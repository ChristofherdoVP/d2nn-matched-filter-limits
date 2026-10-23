"""
76_mascara_branqueada.py — Uma máscara de amplitude e fase PROJETADA realiza o filtro branqueado?

Motivação (script 75): contra o ótimo branqueado, a D²NN 74 (amplitude e fase, treinada a partir
de H = S*) fica em 1,15 no CRISP, e a 67 (só fase) em 1,05. Mas num 4f o campo no pico de
correlação é ⟨h, X⟩, com h a resposta impulsional da máscara: uma máscara de amplitude e fase
realiza QUALQUER peso, inclusive o branqueado w(λ) = t(λ)/σ²(λ). Se isso valer na prática, a
falta de branqueamento do 74 é do treino (inicialização em S*), não da arquitetura.

Teste: só detecção a 5% de falso alarme, sem treino (como o 73). Máscaras calculadas,
H = conj(F{X_w})/max|F{X_w}| na pupila, sem diafragma, leitura pela energia na janela 3 × 5:
  casado S* (instrumento)      : w = t_k (controle; o 73 deu 1,00 no ViSP)
  branqueado (instrumento)     : w = peso na grade LAM que reproduz exatamente o filtro branqueado
                                 da grade nativa, u = t/σ² com σ do Sol quieto degradado
                                 (w = M(MᵀM)⁻¹u, M a interpolação nativa → LAM; IBIS e ViSP: σ
                                 constante → filtro casado nativo)
  branqueado único (3 instr.)  : w = Σ_k p_k·w_k·‖t_k‖/‖w_k‖ (com σ constante reduz à mistura
                                 de templates do 74)
Referências lineares (digitais): filtro |a| (LAM), branqueado na grade LAM e o ótimo branqueado
na grade nativa (script 75). Mesmos conjuntos e sementes do 54/70/73/75; estimador físico (β = 1);
IC 95% por bootstrap pareado em k = 1. Funções de geração, campo e estatística vêm do script 75.

Critérios (definidos antes de rodar), pela razão mediana em k:
  Controle  casado S* (instrumento) / filtro |a| dentro de 1 ± 0,03 nos três instrumentos
            (senão a leitura ou a codificação perdem sinal e o resto não se interpreta)
  W1        branqueado LAM (digital) / ótimo nativo no CRISP ≤ 1,03 (o peso fixo, com σ do Sol
            quieto, quase não perde para o ótimo com σ de cada perfil)
  W2        branqueado (instrumento) / ótimo no CRISP ≤ 1,05 → a máscara de amplitude e fase
            realiza o ótimo; a falta de branqueamento do 74 é do treino
  W3        branqueado único: só reportado (razão contra o ótimo nos três instrumentos)

Uso:
    python Scripts/76_mascara_branqueada.py
"""

import sys
import time
import importlib.util
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from Src.config import DEVICE, DTYPE
from Src.model import D2NNOptica, PUPIL_T
from Src.physics import LAM, V_CORE_KMS, get_stokes, degradar_stokes, reamostrar_para_LAM
from Src.dataset import INST_NOMES, INST_PROBS
from Src.train_utils import X_SCALE_TREINO
from Src.evaluation import templates_filtro_casado, amostrar_varredura_bf, carregar_modelo, MontadorEntrada
from Src.report_utils import save_report

_spec = importlib.util.spec_from_file_location('s75', Path(__file__).with_name('75_referencias_artigo_b.py'))
s75 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s75)

INSTR, FAP, K_GRADE = s75.INSTR, s75.FAP, s75.K_GRADE
JANELA = (3, 5)
D_FILTRO, D_OTIMO = 'filtro |a| (LAM)', 'ótimo branqueado (nativo)'
D_WLAM = 'branqueado LAM (digital)'
D_CASADO, D_BRANQ, D_UNICO = 'casado S* (instrumento)', 'branqueado (instrumento)', 'branqueado único (3 instr.)'


def pesos_branqueados_lam():
    """
    Peso h na grade LAM que reproduz EXATAMENTE o filtro branqueado da grade nativa.
    V_LAM = M·V_nat (M: interpolação linear com V = 0 fora da faixa, como reamostrar_para_LAM).
    O ótimo nativo é u·V_nat com u = t_nat/σ_nat² (σ do Sol quieto, B = 0, degradado sem ruído).
    h = M (MᵀM)⁻¹ u satisfaz Mᵀh = u, então h·V_LAM = u·V_nat para qualquer V_nat.
    (Um peso ingênuo t/σ² na grade LAM não é o ótimo: a interpolação pondera cada ponto nativo
    pelo espaçamento local, e a grade do CRISP é irregular.)
    """
    st0 = get_stokes(0.0, v_km_s=V_CORE_KMS)
    st1 = get_stokes(100.0, filling=1.0, v_km_s=V_CORE_KMS)
    out = {}
    for k, n in enumerate(INST_NOMES):
        ins = INSTR[n]
        d0 = degradar_stokes(st0, LAM, ins, rng_local=np.random.default_rng(0), fator_ruido=0.0)
        d1 = degradar_stokes(st1, LAM, ins, rng_local=np.random.default_rng(0), fator_ruido=0.0)
        lam_obs = d0['lam_obs']
        r = ins['ruido']
        sig = (np.full(len(lam_obs), float(r['sigma'])) if r['tipo'] == 'constante'
               else float(r['eps']) * np.abs(d0['I'].astype(np.float64)))
        u = d1['V'].astype(np.float64) / 100.0 / sig ** 2
        M = np.stack([np.interp(LAM, lam_obs, e, left=0.0, right=0.0) for e in np.eye(len(lam_obs))], axis=1)
        h = M @ np.linalg.solve(M.T @ M, u)
        assert np.allclose(M.T @ h, u, rtol=1e-6, atol=1e-9 * np.abs(u).max())
        out[k] = h
    return out


def mascara(montar, w):
    """H = conj(F{X_w})/max|F{X_w}| na pupila, num contêiner D2NNOptica (janela 3 × 5, sem diafragma)."""
    Xw = montar(torch.tensor(np.asarray(w)[None, :], dtype=DTYPE, device=DEVICE))[0]
    S = torch.fft.fft2(Xw.to(torch.complex64), norm='ortho')
    H = torch.fft.fftshift(torch.conj(S) / S.abs().max(), dim=(-2, -1))
    m = D2NNOptica(torch.zeros_like(PUPIL_T, dtype=DTYPE), JANELA, treinar_fase=False,
                   diafragma_pm=None).to(DEVICE).eval()
    Hp = (H * m.mascara.pupila).to(torch.complex64)
    m.mascara.get_mascara = lambda: Hp
    pup = PUPIL_T.abs() > 0
    return m, float((Hp.abs()[pup] ** 2).mean())


def lin(V, w, t):
    """Estimativa linear de B·f com peso w (normalizada para dar B·f com o template t)."""
    return V @ w / (t @ w)


def main():
    t0 = time.time()
    m67, meta = carregar_modelo(s75.MODELO_67, DEVICE)       # só para o X_scale da entrada
    montar = MontadorEntrada(float(meta.get('X_scale', X_SCALE_TREINO)), DEVICE)
    del m67
    templ = templates_filtro_casado(INST_NOMES, instrumental=INSTR)
    tn = s75.templates_nativos()
    w = pesos_branqueados_lam()

    B0 = np.zeros(1500)
    nulo = s75.conjunto(B0, B0, s75.SEED_NULO_DEG)
    Bv, Fv, _ = amostrar_varredura_bf(1000, s75.SEED_VAR, classes=(0, 1))
    bf_all = Bv * Fv
    var = s75.conjunto(Bv, Fv, s75.SEED_VAR_DEG)
    i0, iv = nulo['inst'], var['inst']
    ot_n, ot_v = s75.amp_otima(nulo, tn), s75.amp_otima(var, tn)
    print(f'Perfis: {len(i0)} sem campo, {len(iv)} na varredura ({time.time() - t0:.0f} s)')

    p = np.asarray(INST_PROBS, dtype=float)
    w_unico = sum(p[k] * w[k] * np.linalg.norm(templ[k]) / np.linalg.norm(w[k]) for k in range(len(INST_NOMES)))
    m_unico, tr_unico = mascara(montar, w_unico)

    dets, transm = {}, {D_UNICO: tr_unico}
    for ki, nm in enumerate(INST_NOMES):
        s0, sv = i0 == ki, iv == ki
        t = templ[ki]
        m_c, tr_c = mascara(montar, t)
        m_b, tr_b = mascara(montar, w[ki])
        transm[f'{D_CASADO} | {nm}'] = tr_c
        transm[f'{D_BRANQ} | {nm}'] = tr_b

        def campos(m):
            return ((s75.campo_janela(m, montar, nulo['V0'][s0]), s75.campo_janela(m, montar, nulo['dV'][s0])),
                    (s75.campo_janela(m, montar, var['V0'][sv]), s75.campo_janela(m, montar, var['dV'][sv])))

        dets[nm] = {
            D_FILTRO: ('lin', (lin(nulo['V0'][s0], t, t), lin(nulo['dV'][s0], t, t)),
                       (lin(var['V0'][sv], t, t), lin(var['dV'][sv], t, t))),
            D_WLAM: ('lin', (lin(nulo['V0'][s0], w[ki], t), lin(nulo['dV'][s0], w[ki], t)),
                     (lin(var['V0'][sv], w[ki], t), lin(var['dV'][sv], w[ki], t))),
            D_OTIMO: ('lin', (ot_n[0][s0], ot_n[1][s0]), (ot_v[0][sv], ot_v[1][sv])),
            D_CASADO: ('energia', *campos(m_c)),
            D_BRANQ: ('energia', *campos(m_b)),
            D_UNICO: ('energia', *campos(m_unico)),
        }
        del m_c, m_b
        print(f'  {nm}: campos calculados ({time.time() - t0:.0f} s)')

    def deteccoes(det, k):
        tipo, (n0, n1), (v0, v1) = det
        e0, ev = s75.escore(tipo, n0, n1, k), s75.escore(tipo, v0, v1, k)
        return (ev > np.quantile(e0, 1 - FAP)).astype(float)

    res = {nm: {d: [] for d in dets[nm]} for nm in INST_NOMES}
    d_k1 = {nm: {} for nm in INST_NOMES}
    for k in K_GRADE:
        for nm in INST_NOMES:
            bf = bf_all[iv == INST_NOMES.index(nm)]
            for d, det in dets[nm].items():
                dd = deteccoes(det, k)
                res[nm][d].append({'k': float(k), 'bf50': s75.bf50(bf, dd)})
                if abs(k - 1.0) < 1e-9:
                    d_k1[nm][d] = dd

    rng = np.random.default_rng(7607)
    pares = [(D_CASADO, D_FILTRO), (D_WLAM, D_OTIMO), (D_FILTRO, D_OTIMO), (D_CASADO, D_OTIMO),
             (D_BRANQ, D_OTIMO), (D_UNICO, D_OTIMO)]
    resumo = {}
    print(f'\n{"="*108}\nB·f₅₀ A 5% DE FALSO ALARME (estimador físico, β = 1) — G\n{"="*108}')
    print(f'  {"instrumento":12s} {"detector":30s}' + ''.join(f'{k:>8.3f}' for k in K_GRADE))
    for nm in INST_NOMES:
        for d in dets[nm]:
            print(f'  {nm:12s} {d:30s}' + ''.join(f'{l["bf50"]:8.1f}' if np.isfinite(l['bf50']) else f'{"—":>8s}'
                                                   for l in res[nm][d]))
    print(f'\n  Razões: mediana em k (faixa); k = 1 [IC 95% pareado]')
    for a, b in pares:
        for ki, nm in enumerate(INST_NOMES):
            r = np.array([la['bf50'] / lb['bf50'] for la, lb in zip(res[nm][a], res[nm][b])])
            _, _, r1, ic = s75.razao_pareada(bf_all[iv == ki], d_k1[nm][a], d_k1[nm][b], rng)
            resumo[f'{a} / {b} | {nm}'] = {'mediana': float(np.nanmedian(r)), 'min': float(np.nanmin(r)),
                                           'max': float(np.nanmax(r)), 'k1': r1, 'k1_ic': ic}
            print(f'  {a:30s} / {b:27s} {nm:12s} {np.nanmedian(r):5.2f} ({np.nanmin(r):.2f}–{np.nanmax(r):.2f})'
                  f'   k = 1: {r1:.2f} [{ic[0]:.2f}, {ic[1]:.2f}]')

    print(f'\n  Transmissão ⟨|H|²⟩ na pupila: ' + ', '.join(f'{n} {100 * v:.2f}%' for n, v in transm.items()))

    med = lambda a, b, nm: resumo[f'{a} / {b} | {nm}']['mediana']
    controle = all(abs(med(D_CASADO, D_FILTRO, nm) - 1) <= 0.03 for nm in INST_NOMES)
    w1 = med(D_WLAM, D_OTIMO, 'CRISP/SST') <= 1.03
    w2 = med(D_BRANQ, D_OTIMO, 'CRISP/SST') <= 1.05
    print(f'\n{"="*108}\nDECISÃO (critérios definidos antes)\n{"="*108}')
    print(f'  Controle (casado S* / filtro |a| em 1 ± 0,03): '
          + ', '.join(f'{nm} {med(D_CASADO, D_FILTRO, nm):.2f}' for nm in INST_NOMES)
          + f' → {"ATENDIDO" if controle else "NÃO ATENDIDO"}')
    print(f'  W1 (branqueado LAM / ótimo, CRISP ≤ 1,03): {med(D_WLAM, D_OTIMO, "CRISP/SST"):.2f} → '
          f'{"ATENDIDO" if w1 else "NÃO ATENDIDO"}')
    print(f'  W2 (máscara branqueada / ótimo, CRISP ≤ 1,05): {med(D_BRANQ, D_OTIMO, "CRISP/SST"):.2f} → '
          f'{"ATENDIDO" if w2 else "NÃO ATENDIDO"}')
    print(f'  W3 (máscara branqueada única / ótimo, só reportado): '
          + ', '.join(f'{nm} {med(D_UNICO, D_OTIMO, nm):.2f}' for nm in INST_NOMES))
    if controle and w2:
        print('  → a máscara de amplitude e fase realiza o ótimo branqueado; a falta de branqueamento do 74 é do treino')
    elif not controle:
        print('  → o controle falhou: há perda na leitura ou na codificação; W2 não se interpreta')
    else:
        print('  → a máscara branqueada não atinge o ótimo: limite da leitura em janela ou da codificação')
    print(f'  Tempo: {(time.time() - t0) / 60:.1f} min')
    save_report('76_mascara_branqueada', {'fap': FAP, 'k_grade': K_GRADE, 'janela': list(JANELA),
                                          'resultados': res, 'resumo': resumo, 'transmissao': transm,
                                          'criterios': {'controle': controle, 'W1': w1, 'W2': w2}})


if __name__ == '__main__':
    main()
