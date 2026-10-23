"""
73_mascara_visp.py — De onde vem a perda da D²NN óptica no ViSP? Máscara por instrumento.

Problema (54 com o 67, estimador físico do 70): razão B·f₅₀ D²NN 67 / filtro |a| no ViSP de
1,38 (mediana em k; 1,20–1,51), contra 1,05 no IBIS e 0,82 no CRISP. Hipóteses:
  H1 — máscara única: o POF do 67 vem do template MÉDIO dos três instrumentos (IBIS e CRISP
       só têm V na faixa medida; o ViSP cobre ±5 Å com a LSF mais estreita). Para o ViSP, o
       POF do template médio não é o casado.
  H2 — limite da máscara só de fase: o POF (H = S*/|S|) tem ganho 1 em todas as frequências;
       o filtro casado é H = S* (amplitude e fase). No ViSP, com a janela inteira coberta de
       ruído, a falta de amplitude pesa mais (achado 24; o diafragma resolveu só em parte).
  H3 — leitura: a janela de 3 × 5 pixels ou a codificação da entrada perdem sinal.

Teste: só DETECÇÃO a 5% de falso alarme, que não precisa de treino — o escore é a energia na
janela, e o limiar sai dos perfis sem campo. As máscaras são fixas (calculadas, não treinadas):
  67                       : o modelo atual (POF do template médio, fase treinada, ±200 pm)
  POF médio ±200 (fixo)    : o POF de partida do 67, sem treino (isola o efeito do treino)
  POF ViSP, D = —/200/120/80: POF do template do ViSP, com diafragma ±D pm (— = sem)
  casado complexo ViSP, —/200: H = S* (amplitude + fase), template do ViSP — o filtro casado
                             feito opticamente; limite superior de uma máscara no plano de Fourier
  filtro |a|               : referência (template do ViSP, todos os pontos)
Todos com a mesma entrada (MontadorEntrada do 67) e a mesma janela (3 × 5).

Dados: só perfis do ViSP (INST_PROBS = ViSP), mesmos B, f e sementes da varredura do 54
(Quieta + Rede, 2000 perfis) e do conjunto sem campo (1500). Ruído exato e rápido: o campo na
janela é linear em V, então E(k) = Σ|c₀ + k·c₁|² com c₀ (sem ruído) e c₁ (só ruído) calculados
uma vez. Estimador físico (β = 1, script 70); razão com IC 95% por bootstrap pareado.

Decisão (definida antes), pela razão mediana em k contra o filtro |a|:
  H1 confirmada se algum POF ViSP ficar ≤ 1,10 → solução: uma máscara por instrumento
     (no artigo: o POF casado ao instrumento).
  H2 confirmada se nenhum POF ViSP ficar ≤ 1,10 mas o casado complexo ficar ≤ 1,05 → a perda é
     da máscara só de fase; solução: máscara de amplitude e fase.
  H3 se nem o casado complexo chegar a 1,05 → a perda está na leitura ou na entrada.

Uso:
    python Scripts/73_mascara_visp.py
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt

from Src.config import DEVICE, DTYPE, FIGURES_DIR
from Src.model import D2NNOptica, PUPIL_T
from Src.physics import INSTRUMENTAL_V2
from Src.dataset import generate_degraded_profiles, INST_NOMES, SEED_BASE
from Src.train_utils import X_SCALE_TREINO, MODEL_INST_V2_PATH
from Src.evaluation import (templates_filtro_casado, amostrar_varredura_bf, carregar_modelo,
                            MontadorEntrada, _ajuste_fisico, limiar_bf50_fisico)
from Src.treino_optico import fase_pof, janela_diafragma
from Src.report_utils import save_report
from scipy.stats import norm

INSTR = INSTRUMENTAL_V2
FAP = 0.05
K_GRADE = sorted({0.1, 0.2, 0.5, 0.7, 1.0, 1.4, 2.0, 1 / np.sqrt(44), 1 / np.sqrt(11)})
SEED_NULO_DEG, SEED_VAR, SEED_VAR_DEG = SEED_BASE + 4003, SEED_BASE + 4002, SEED_BASE + 4004
N_BOOT = 200
JANELA = (3, 5)
MODELO_67 = MODEL_INST_V2_PATH.with_name('modelo_d2nn_optico_v2.pt')
I_VISP = INST_NOMES.index('ViSP/DKIST')
P_VISP = [1.0 if n == 'ViSP/DKIST' else 0.0 for n in INST_NOMES]
Z_C = norm.isf(FAP / 2)
Z50 = 1.959963984540054


def conjunto(B, F, seed):
    _, V0, _ = generate_degraded_profiles(B, F, seed=seed, inst_probs=P_VISP, fator_ruido=0.0, instrumental=INSTR)
    _, V1, _ = generate_degraded_profiles(B, F, seed=seed, inst_probs=P_VISP, fator_ruido=1.0, instrumental=INSTR)
    return V0, V1 - V0


def campo_janela(m, montar, V, lote=256):
    """Campo complexo nos pixels da janela (linear em V): mesma conta de D2NNOptica.energia sem o |·|²."""
    out = []
    with torch.no_grad():
        for i in range(0, len(V), lote):
            x = montar(torch.tensor(V[i:i + lote], dtype=DTYPE, device=DEVICE))
            if m.diafragma is not None:
                x = x * m.diafragma
            E = m.mascara(x, aplicar_pad=False)
            ny, nx = E.shape[-2], E.shape[-1]
            E = torch.roll(E, shifts=(ny // 2, nx // 2), dims=(-2, -1))
            hy, hx = m.janela[0] // 2, m.janela[1] // 2
            w = E[..., ny // 2 - hy: ny // 2 + hy + 1, nx // 2 - hx: nx // 2 + hx + 1]
            out.append(w.reshape(len(x), -1).cpu().numpy())
    return np.concatenate(out)


def espectro(montar, t):
    Xt = montar(torch.tensor(t[None, :], dtype=DTYPE, device=DEVICE))[0]
    return torch.fft.fft2(Xt.to(torch.complex64), norm='ortho')


def mascara_pof(montar, t, D):
    fase = torch.fft.fftshift(-torch.angle(espectro(montar, t * janela_diafragma(D))), dim=(-2, -1)).to(DTYPE)
    return D2NNOptica(fase, JANELA, treinar_fase=False, diafragma_pm=D).to(DEVICE).eval()


def mascara_complexa(montar, t, D):
    """H = S* (amplitude normalizada ao máximo + fase), na mesma pupila e com o mesmo diafragma."""
    S = espectro(montar, t * janela_diafragma(D))
    H = torch.fft.fftshift(torch.conj(S) / S.abs().max(), dim=(-2, -1))
    m = D2NNOptica(torch.zeros_like(PUPIL_T.real if PUPIL_T.is_complex() else PUPIL_T, dtype=DTYPE),
                   JANELA, treinar_fase=False, diafragma_pm=D).to(DEVICE).eval()
    pup = m.mascara.pupila
    Hp = (H * pup).to(torch.complex64)
    m.mascara.get_mascara = lambda: Hp
    return m


def razao_pareada(bf, d_det, d_ref, rng):
    """B·f₅₀ (físico, β = 1) de cada detector e a razão, com IC 95% por bootstrap pareado."""
    e_d = limiar_bf50_fisico(bf, d_det, None, n_boot=0, fap=FAP)[0]
    e_r = limiar_bf50_fisico(bf, d_ref, None, n_boot=0, fap=FAP)[0]
    if not (np.isfinite(e_d) and np.isfinite(e_r)):
        return e_d, e_r, np.nan, [np.nan, np.nan]
    rs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(bf), len(bf))
        if d_det[i].min() == d_det[i].max() or d_ref[i].min() == d_ref[i].max():
            continue
        sd = np.exp(_ajuste_fisico(bf[i], d_det[i], Z_C, False)[0])
        sr = np.exp(_ajuste_fisico(bf[i], d_ref[i], Z_C, False)[0])
        rs.append(sd / sr)
    ic = [float(v) for v in np.percentile(rs, [2.5, 97.5])] if len(rs) >= N_BOOT // 2 else [np.nan, np.nan]
    return e_d, e_r, e_d / e_r, ic


def main():
    t0 = time.time()
    if not MODELO_67.exists():
        sys.exit(f'{MODELO_67} não encontrado (rode o script 67).')
    m67, meta = carregar_modelo(MODELO_67, DEVICE)
    montar = MontadorEntrada(float(meta.get('X_scale', X_SCALE_TREINO)), DEVICE)
    templ = templates_filtro_casado(INST_NOMES, instrumental=INSTR)
    t_visp = templ[I_VISP]

    variantes = {'67 (atual)': m67,
                 'POF médio ±200 (fixo)': D2NNOptica(fase_pof(montar, 200), JANELA, treinar_fase=False,
                                                     diafragma_pm=200).to(DEVICE).eval()}
    for D in (None, 200, 120, 80):
        variantes[f'POF ViSP {"±" + str(D) if D else "sem diafr."}'] = mascara_pof(montar, t_visp, D)
    for D in (None, 200):
        variantes[f'casado complexo ViSP {"±" + str(D) if D else "sem diafr."}'] = mascara_complexa(montar, t_visp, D)

    # Verificação: o campo na janela reproduz a energia do modelo
    Vt = generate_degraded_profiles(np.full(8, 300.0), np.full(8, 0.3), seed=1, inst_probs=P_VISP,
                                    instrumental=INSTR)[1]
    with torch.no_grad():
        e_mod = m67.energia(montar(torch.tensor(Vt, dtype=DTYPE, device=DEVICE))).cpu().numpy()
    e_cj = (np.abs(campo_janela(m67, montar, Vt)) ** 2).sum(1)
    assert np.allclose(e_mod, e_cj, rtol=1e-3), 'campo_janela não reproduz D2NNOptica.energia'

    B0 = np.zeros(1500)
    V00, dV0 = conjunto(B0, B0, SEED_NULO_DEG)
    Bv, Fv, _ = amostrar_varredura_bf(1000, SEED_VAR, classes=(0, 1))
    bf = Bv * Fv
    Vv0, dVv = conjunto(Bv, Fv, SEED_VAR_DEG)
    print(f'Perfis do ViSP: {len(V00)} sem campo, {len(Vv0)} na varredura ({time.time() - t0:.0f} s)')

    campos = {}
    for nome, m in variantes.items():
        campos[nome] = (campo_janela(m, montar, V00), campo_janela(m, montar, dV0),
                        campo_janela(m, montar, Vv0), campo_janela(m, montar, dVv))
    tt = t_visp / (t_visp @ t_visp)
    a = (V00 @ tt, dV0 @ tt, Vv0 @ tt, dVv @ tt)
    print(f'Campos calculados ({time.time() - t0:.0f} s)')

    rng = np.random.default_rng(7307)
    res = {nome: [] for nome in variantes}
    for k in K_GRADE:
        e0_f, ev_f = np.abs(a[0] + k * a[1]), np.abs(a[2] + k * a[3])
        d_f = (ev_f > np.quantile(e0_f, 1 - FAP)).astype(float)
        for nome, (c00, c01, cv0, cv1) in campos.items():
            e0 = (np.abs(c00 + k * c01) ** 2).sum(1)
            ev = (np.abs(cv0 + k * cv1) ** 2).sum(1)
            d = (ev > np.quantile(e0, 1 - FAP)).astype(float)
            e_d, e_r, r, ic = razao_pareada(bf, d, d_f, rng)
            res[nome].append({'k': float(k), 'bf50': e_d, 'bf50_filtro': e_r, 'razao': r, 'ic': ic})
        print(f'  k = {k:.3f}: concluído ({time.time() - t0:.0f} s)')

    print(f'\n{"="*100}\nViSP — RAZÃO B·f₅₀ / filtro |a| a 5% de falso alarme (estimador físico; IC 95% pareado)\n{"="*100}')
    cab = ''.join(f'{k:>8.3f}' for k in K_GRADE)
    print(f'  {"variante":32s}{cab}   mediana   k = 1 [IC 95%]')
    f_k1 = [l for l in res['67 (atual)'] if abs(l['k'] - 1) < 1e-9][0]['bf50_filtro']
    print(f'  {"(B·f₅₀ do filtro, G)":32s}' + ''.join(f'{l["bf50_filtro"]:8.1f}' for l in res['67 (atual)']))
    resumo = {}
    for nome, ls in res.items():
        r = np.array([l['razao'] for l in ls])
        med = float(np.nanmedian(r))
        l1 = [l for l in ls if abs(l['k'] - 1) < 1e-9][0]
        resumo[nome] = {'mediana': med, 'k1': l1['razao'], 'k1_ic': l1['ic']}
        print(f'  {nome:32s}' + ''.join(f'{v:8.2f}' if np.isfinite(v) else f'{"—":>8s}' for v in r)
              + f'   {med:7.2f}   {l1["razao"]:.2f} [{l1["ic"][0]:.2f}, {l1["ic"][1]:.2f}]')

    pof = {n: v['mediana'] for n, v in resumo.items() if n.startswith('POF ViSP')}
    cpx = {n: v['mediana'] for n, v in resumo.items() if n.startswith('casado complexo')}
    melhor_pof, melhor_cpx = min(pof, key=pof.get), min(cpx, key=cpx.get)
    print(f'\n{"="*100}\nDECISÃO (definida antes)\n{"="*100}')
    print(f'  melhor POF ViSP: {melhor_pof} ({pof[melhor_pof]:.2f}) | melhor casado complexo: '
          f'{melhor_cpx} ({cpx[melhor_cpx]:.2f}) | 67: {resumo["67 (atual)"]["mediana"]:.2f}')
    if pof[melhor_pof] <= 1.10:
        decisao = 'H1 — máscara única; solução: uma máscara (POF) por instrumento'
    elif cpx[melhor_cpx] <= 1.05:
        decisao = 'H2 — limite da máscara só de fase; solução: máscara de amplitude e fase'
    else:
        decisao = 'H3 — a perda está na leitura (janela) ou na entrada'
    print(f'  → {decisao}')

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for nome, ls in res.items():
        estilo = '-' if nome.startswith('67') else ('--' if nome.startswith('POF ViSP') else ':')
        ax.plot([l['k'] for l in ls], [l['razao'] for l in ls], estilo, marker='o', ms=3, label=nome)
    ax.axhline(1, color='k', lw=0.6)
    ax.set_xscale('log'); ax.set_xlabel('k (ruído / ruído real)'); ax.set_ylabel('B·f₅₀ / filtro |a| (ViSP)')
    ax.legend(fontsize=7, ncol=2)
    fig_path = Path(FIGURES_DIR) / 'cel73_mascara_visp.png'
    fig.savefig(fig_path, dpi=150)
    print(f'\n  Figura: {fig_path.name}   Tempo: {(time.time() - t0) / 60:.1f} min')
    save_report('73_mascara_visp', {'k_grade': K_GRADE, 'fap': FAP, 'janela': JANELA, 'n_boot': N_BOOT,
                                    'resultados': res, 'resumo': resumo, 'decisao': decisao,
                                    'bf50_filtro_k1': f_k1})


if __name__ == '__main__':
    main()
