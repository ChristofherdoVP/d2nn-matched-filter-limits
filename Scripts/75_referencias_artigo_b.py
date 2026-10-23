"""
75_referencias_artigo_b.py — Três cálculos para o artigo B (Applied Optics), sem treino.

  A. Ótimo branqueado contra k. O filtro |a| de referência (54, 67, 73, 74) projeta V na grade
     LAM com pesos iguais. Com ruído ∝ I (CRISP, eps·I local) ele não é o ótimo. O detector
     ótimo de uma amplitude com forma conhecida e ruído gaussiano independente por ponto é o
     filtro casado BRANQUEADO na grade NATIVA do instrumento:
         a_w = Σ t_k V_k / σ_k²  /  Σ t_k² / σ_k²
     (σ_k do perfil sem ruído; no IBIS e no ViSP, σ constante → filtro casado nativo comum,
     que serve de controle: razão ótimo/filtro ≈ 1). Detecção a 5% de falso alarme com o
     estimador físico (β = 1, script 70), D²NN 67 e 74 lidas pela energia na janela (como no 73).
  B. Fração de sinal e de ruído transmitida pelas máscaras (67 e 74): energia que passa pela
     máscara e energia que chega à janela, relativas à energia na entrada (antes do diafragma).
     Sinal: template de cada instrumento (B·f = 100 G). Ruído: 300 realizações por instrumento.
  C. Quantização de fase do SLM: a fase treinada (67 e 74) requantizada em n bits (níveis
     uniformes em [−π, π)), sem retreino; no 74 também uma variante com a amplitude quantizada
     em 2ⁿ níveis em [0, 1] (só reportada).

Dados: mesmos conjuntos e sementes do 54/70/73 (sem campo 1500; varredura Quieta + Rede,
1000 por classe; mistura de instrumentos INST_PROBS). Os perfis são gerados com a mesma cadeia
de generate_degraded_profiles, guardando também a grade nativa (verificado contra a função).
Ruído exato e rápido: tudo é linear em V, então a(k) = a₀ + k·a₁ e campo(k) = c₀ + k·c₁.

Critérios (definidos antes de rodar):
  A  CRISP, k = 1: a D²NN 67 ATINGE o ótimo se a razão 67/ótimo tiver IC 95% contendo 1 ou
     for ≤ 1,05; caso contrário, o texto diz que ela SE APROXIMA do ótimo.
  B  só reportado.
  C  bits suficientes = o menor n tal que, para todo n' ≥ n testado, a perda de B·f₅₀ em k = 1
     contra a máscara contínua é ≤ 5% (razão ≤ 1,05) nos três instrumentos. Por modelo.

Uso:
    python Scripts/75_referencias_artigo_b.py
"""

import sys
import copy
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm

from Src.config import DEVICE, DTYPE, FIGURES_DIR
from Src.model import PUPIL_T
from Src.physics import INSTRUMENTAL_V2, LAM, V_CORE_KMS, get_stokes, degradar_stokes, reamostrar_para_LAM
from Src.dataset import generate_degraded_profiles, INST_NOMES, INST_PROBS, SEED_BASE
from Src.train_utils import X_SCALE_TREINO, MODEL_INST_V2_PATH
from Src.evaluation import (templates_filtro_casado, amostrar_varredura_bf, carregar_modelo,
                            MontadorEntrada, _ajuste_fisico, limiar_bf50_fisico)
from Src.report_utils import save_report

INSTR = INSTRUMENTAL_V2
FAP = 0.05
Z_C = norm.isf(FAP / 2)
K_GRADE = sorted({0.1, 0.2, 0.5, 0.7, 1.0, 1.4, 2.0, 1 / np.sqrt(44), 1 / np.sqrt(11)})
SEED_NULO_DEG, SEED_VAR, SEED_VAR_DEG = SEED_BASE + 4003, SEED_BASE + 4002, SEED_BASE + 4004
N_BOOT = 200
BITS = (8, 6, 5, 4, 3, 2)
N_RUIDO_FRAC = 300
CKPT = MODEL_INST_V2_PATH.parent
MODELO_67 = CKPT / 'modelo_d2nn_optico_v2.pt'
MODELO_74 = CKPT / 'modelo_d2nn_complexo_v2.pt'
D_FILTRO, D_OTIMO = 'filtro |a| (LAM)', 'ótimo branqueado (nativo)'
D_67, D_74 = 'D²NN 67 (só fase)', 'D²NN 74 (amp. e fase)'


# ─────────────────────────── geração (com grade nativa) ───────────────────────────
def gerar(B, F, seed, fator):
    """Mesma cadeia e mesma sequência aleatória de generate_degraded_profiles; guarda também
    V e σ na grade nativa do instrumento."""
    rng = np.random.default_rng(seed)
    N = len(B)
    V = np.empty((N, len(LAM)))
    inst = np.empty(N, dtype=np.int64)
    Vn, sn = [], []
    for i in range(N):
        k = int(rng.choice(len(INST_NOMES), p=INST_PROBS))
        inst[i] = k
        st = get_stokes(float(B[i]), filling=float(F[i]), v_km_s=V_CORE_KMS)
        instr = INSTR[INST_NOMES[k]]
        deg = degradar_stokes(st, LAM, instr, rng_local=rng, fator_ruido=fator)
        V[i] = reamostrar_para_LAM(deg, instr)[1]
        Vn.append(deg['V'].astype(np.float64))
        sn.append(np.asarray(deg['sigma_por_ponto'], dtype=np.float64))
    return V, inst, Vn, sn


def conjunto(B, F, seed):
    V0, i0, Vn0, _ = gerar(B, F, seed, 0.0)
    V1, i1, Vn1, sn1 = gerar(B, F, seed, 1.0)
    assert (i0 == i1).all()
    _, Vref, iref = generate_degraded_profiles(B[:20], F[:20], seed=seed, instrumental=INSTR)
    assert (iref == i1[:20]).all() and np.allclose(Vref, V1[:20]), \
        'a geração local não reproduz generate_degraded_profiles'
    return {'V0': V0, 'dV': V1 - V0, 'inst': i0, 'Vn0': Vn0,
            'dVn': [b - a for a, b in zip(Vn0, Vn1)], 'sig': sn1}


def templates_nativos():
    """Forma de V por G de B·f na grade nativa (mesma cadeia, sem ruído)."""
    st = get_stokes(100.0, filling=1.0, v_km_s=V_CORE_KMS)
    rng = np.random.default_rng(0)
    return {k: degradar_stokes(st, LAM, INSTR[n], rng_local=rng, fator_ruido=0.0)['V'].astype(np.float64) / 100.0
            for k, n in enumerate(INST_NOMES)}


# ─────────────────────────── estatísticas de detecção ───────────────────────────
def amp_lam(c, templ):
    """Filtro |a| de referência (grade LAM, pesos iguais): partes sem ruído e só ruído."""
    a0, a1 = np.empty(len(c['inst'])), np.empty(len(c['inst']))
    for k in range(len(INST_NOMES)):
        m = c['inst'] == k
        t = templ[k]
        a0[m] = c['V0'][m] @ t / (t @ t)
        a1[m] = c['dV'][m] @ t / (t @ t)
    return a0, a1


def amp_otima(c, tn):
    """Filtro casado branqueado na grade nativa (pesos 1/σ² do perfil sem ruído)."""
    a0, a1 = np.empty(len(c['inst'])), np.empty(len(c['inst']))
    for i, k in enumerate(c['inst']):
        t = tn[int(k)]
        w = 1.0 / np.maximum(c['sig'][i], 1e-15) ** 2
        den = np.sum(w * t * t)
        a0[i] = np.sum(w * t * c['Vn0'][i]) / den
        a1[i] = np.sum(w * t * c['dVn'][i]) / den
    return a0, a1


def campo_janela(m, montar, V, lote=256):
    """Campo complexo nos pixels da janela (linear em V) — como no script 73."""
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


def escore(tipo, p0, p1, k):
    return np.abs(p0 + k * p1) if tipo == 'lin' else (np.abs(p0 + k * p1) ** 2).sum(1)


def deteccoes(det, k, ki, i0, iv):
    tipo, (n0, n1), (v0, v1) = det
    e0, ev = escore(tipo, n0, n1, k), escore(tipo, v0, v1, k)
    thr = np.quantile(e0[i0 == ki], 1 - FAP)
    return (ev[iv == ki] > thr).astype(float)


def bf50(bf, d):
    return limiar_bf50_fisico(bf, d, None, n_boot=0, fap=FAP)[0]


def razao_pareada(bf, d_a, d_b, rng):
    """B·f₅₀ de dois detectores nos MESMOS perfis e a razão a/b, com IC 95% por bootstrap pareado."""
    ea, eb = bf50(bf, d_a), bf50(bf, d_b)
    if not (np.isfinite(ea) and np.isfinite(eb)):
        return ea, eb, np.nan, [np.nan, np.nan]
    rs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(bf), len(bf))
        if d_a[i].min() == d_a[i].max() or d_b[i].min() == d_b[i].max():
            continue
        sa = np.exp(_ajuste_fisico(bf[i], d_a[i], Z_C, False)[0])
        sb = np.exp(_ajuste_fisico(bf[i], d_b[i], Z_C, False)[0])
        rs.append(sa / sb)
    ic = [float(v) for v in np.percentile(rs, [2.5, 97.5])] if len(rs) >= N_BOOT // 2 else [np.nan, np.nan]
    return ea, eb, ea / eb, ic


# ─────────────────────────── quantização ───────────────────────────
def wrap(phi):
    return torch.remainder(phi + np.pi, 2 * np.pi) - np.pi


def modelo_quantizado(m, bits, quant_amp=False):
    mq = copy.deepcopy(m)
    cam = mq.mascara
    q = 2 * np.pi / 2 ** bits
    with torch.no_grad():
        phi = cam.fase_base + cam.delta_phi
        cam.delta_phi.copy_(torch.round(wrap(phi) / q) * q - cam.fase_base)
        if quant_amp and hasattr(cam, 'rho'):
            L = 2 ** bits - 1
            aq = torch.clamp(torch.round(torch.sigmoid(cam.rho) * L) / L, 1e-6, 1 - 1e-6)
            cam.rho.copy_(torch.log(aq / (1 - aq)))
    return mq.eval()


# ─────────────────────────── frações transmitidas ───────────────────────────
def fracoes(m, montar, V, lote=128):
    """Energia na entrada (antes do diafragma), depois da máscara e na janela — somadas no lote."""
    ein = eout = ewin = 0.0
    with torch.no_grad():
        for i in range(0, len(V), lote):
            x = montar(torch.tensor(V[i:i + lote], dtype=DTYPE, device=DEVICE))
            ein += float((x.abs() ** 2).sum())
            xd = x * m.diafragma if m.diafragma is not None else x
            eout += float((m.mascara(xd, aplicar_pad=False).abs() ** 2).sum())
    ewin = float((np.abs(campo_janela(m, montar, V)) ** 2).sum())
    return {'passa': eout / ein, 'janela': ewin / ein}


def main():
    t0 = time.time()
    for p in (MODELO_67, MODELO_74):
        if not p.exists():
            sys.exit(f'{p} não encontrado (rode os scripts 67 e 74).')
    m67, meta67 = carregar_modelo(MODELO_67, DEVICE)
    m74, meta74 = carregar_modelo(MODELO_74, DEVICE)
    xs = float(meta67.get('X_scale', X_SCALE_TREINO))
    assert abs(float(meta74.get('X_scale', X_SCALE_TREINO)) - xs) < 1e-12
    montar = MontadorEntrada(xs, DEVICE)
    templ = templates_filtro_casado(INST_NOMES, instrumental=INSTR)
    tn = templates_nativos()

    # Verificação: o campo na janela reproduz a energia de cada modelo
    Vt = generate_degraded_profiles(np.full(8, 300.0), np.full(8, 0.3), seed=1, instrumental=INSTR)[1]
    for m in (m67, m74):
        with torch.no_grad():
            e_mod = m.energia(montar(torch.tensor(Vt, dtype=DTYPE, device=DEVICE))).cpu().numpy()
        assert np.allclose(e_mod, (np.abs(campo_janela(m, montar, Vt)) ** 2).sum(1), rtol=1e-3)

    B0 = np.zeros(1500)
    nulo = conjunto(B0, B0, SEED_NULO_DEG)
    Bv, Fv, _ = amostrar_varredura_bf(1000, SEED_VAR, classes=(0, 1))
    bf_all = Bv * Fv
    var = conjunto(Bv, Fv, SEED_VAR_DEG)
    i0, iv = nulo['inst'], var['inst']
    print(f'Perfis: {len(i0)} sem campo, {len(iv)} na varredura; geração conferida ({time.time() - t0:.0f} s)')

    def campos(m, c):
        return campo_janela(m, montar, c['V0']), campo_janela(m, montar, c['dV'])

    dets = {D_FILTRO: ('lin', amp_lam(nulo, templ), amp_lam(var, templ)),
            D_OTIMO: ('lin', amp_otima(nulo, tn), amp_otima(var, tn)),
            D_67: ('energia', campos(m67, nulo), campos(m67, var)),
            D_74: ('energia', campos(m74, nulo), campos(m74, var))}
    print(f'Estatísticas calculadas ({time.time() - t0:.0f} s)')

    # ═════════════════════════ A. ótimo branqueado contra k ═════════════════════════
    resA = {d: {nm: [] for nm in INST_NOMES} for d in dets}
    det_k1 = {d: {} for d in dets}
    for k in K_GRADE:
        for ki, nm in enumerate(INST_NOMES):
            bf = bf_all[iv == ki]
            for d, det in dets.items():
                dd = deteccoes(det, k, ki, i0, iv)
                resA[d][nm].append({'k': float(k), 'bf50': bf50(bf, dd)})
                if abs(k - 1.0) < 1e-9:
                    det_k1[d][nm] = dd
    rng = np.random.default_rng(7507)
    pares = [(D_OTIMO, D_FILTRO), (D_67, D_OTIMO), (D_74, D_OTIMO), (D_67, D_FILTRO), (D_74, D_FILTRO)]
    pares_k1 = {}
    for a, b in pares:
        for ki, nm in enumerate(INST_NOMES):
            ea, eb, r, ic = razao_pareada(bf_all[iv == ki], det_k1[a][nm], det_k1[b][nm], rng)
            pares_k1[f'{a} / {b} | {nm}'] = {'bf50_a': ea, 'bf50_b': eb, 'razao': r, 'ic': ic}

    def razoes_k(a, b, nm):
        return np.array([la['bf50'] / lb['bf50'] for la, lb in zip(resA[a][nm], resA[b][nm])])

    print(f'\n{"="*104}\nA. B·f₅₀ A 5% DE FALSO ALARME (estimador físico, β = 1) — G\n{"="*104}')
    print(f'  {"instrumento":12s} {"detector":28s}' + ''.join(f'{k:>8.3f}' for k in K_GRADE))
    for nm in INST_NOMES:
        for d in dets:
            print(f'  {nm:12s} {d:28s}' + ''.join(f'{l["bf50"]:8.1f}' if np.isfinite(l['bf50']) else f'{"—":>8s}'
                                                   for l in resA[d][nm]))
    print(f'\n  Razões (mediana em k; faixa) e em k = 1 com IC 95% pareado:')
    resumoA = {}
    for a, b in pares:
        for nm in INST_NOMES:
            r = razoes_k(a, b, nm)
            p1 = pares_k1[f'{a} / {b} | {nm}']
            chave = f'{a} / {b} | {nm}'
            resumoA[chave] = {'mediana': float(np.nanmedian(r)), 'min': float(np.nanmin(r)),
                              'max': float(np.nanmax(r)), 'k1': p1}
            print(f'  {a:28s} / {b:28s} {nm:12s} {np.nanmedian(r):5.2f} ({np.nanmin(r):.2f}–{np.nanmax(r):.2f})'
                  f'   k = 1: {p1["razao"]:.2f} [{p1["ic"][0]:.2f}, {p1["ic"][1]:.2f}]')

    c67 = pares_k1[f'{D_67} / {D_OTIMO} | CRISP/SST']
    atinge = bool(np.isfinite(c67['razao']) and
                  ((c67['ic'][0] <= 1.0 <= c67['ic'][1]) or c67['razao'] <= 1.05))
    print(f'\n  Critério A (CRISP, k = 1): 67/ótimo = {c67["razao"]:.2f} [{c67["ic"][0]:.2f}, {c67["ic"][1]:.2f}]'
          f' → {"ATINGE o ótimo" if atinge else "SE APROXIMA do ótimo (não atinge)"}')
    print(f'  B·f₅₀ do ótimo no CRISP em k = 1: {c67["bf50_b"]:.1f} G (filtro |a|: '
          f'{pares_k1[f"{D_OTIMO} / {D_FILTRO} | CRISP/SST"]["bf50_b"]:.1f} G)')

    # ═════════════════════════ B. frações transmitidas ═════════════════════════
    resB = {}
    print(f'\n{"="*104}\nB. FRAÇÃO DA ENERGIA DE ENTRADA QUE PASSA PELA MÁSCARA E QUE CHEGA À JANELA\n{"="*104}')
    pup = PUPIL_T.abs() > 0
    t74 = float((m74.mascara.amplitude().detach()[pup] ** 2).mean())
    print(f'  Transmissão média ⟨a²⟩ da máscara 74 na pupila: {100 * t74:.2f}%  (67: 100% na pupila)')
    print(f'  {"modelo":24s} {"instrumento":12s} {"sinal passa":>12s} {"sinal janela":>13s} '
          f'{"ruído passa":>12s} {"ruído janela":>13s} {"(s/r) janela":>13s}')
    for nome, m in ((D_67, m67), (D_74, m74)):
        resB[nome] = {}
        for ki, nm in enumerate(INST_NOMES):
            fs = fracoes(m, montar, (templ[ki] * 100.0)[None, :])
            dv = nulo['dV'][i0 == ki][:N_RUIDO_FRAC]
            fr = fracoes(m, montar, dv)
            resB[nome][nm] = {'sinal': fs, 'ruido': fr}
            print(f'  {nome:24s} {nm:12s} {100 * fs["passa"]:11.2f}% {100 * fs["janela"]:12.3f}% '
                  f'{100 * fr["passa"]:11.2f}% {100 * fr["janela"]:12.4f}% {fs["janela"] / fr["janela"]:13.1f}')

    # ═════════════════════════ C. quantização de fase ═════════════════════════
    print(f'\n{"="*104}\nC. QUANTIZAÇÃO DA MÁSCARA — razão B·f₅₀(quantizada)/B·f₅₀(contínua) em k = 1 [IC 95%]\n{"="*104}')
    variantes = [(D_67, m67, False, '67 fase'), (D_74, m74, False, '74 fase'), (D_74, m74, True, '74 fase + amp.')]
    resC = {}
    for base, m, qa, rot in variantes:
        resC[rot] = {}
        for bits in BITS:
            mq = modelo_quantizado(m, bits, quant_amp=qa)
            detq = ('energia', campos(mq, nulo), campos(mq, var))
            resC[rot][bits] = {}
            for ki, nm in enumerate(INST_NOMES):
                dq = deteccoes(detq, 1.0, ki, i0, iv)
                eq, ec, r, ic = razao_pareada(bf_all[iv == ki], dq, det_k1[base][nm], rng)
                resC[rot][bits][nm] = {'bf50': eq, 'bf50_continua': ec, 'razao': r, 'ic': ic}
            del mq
            print(f'  {rot:16s} {bits} bits: ' + '   '.join(
                f'{nm} {v["razao"]:.3f} [{v["ic"][0]:.2f}, {v["ic"][1]:.2f}]' for nm, v in resC[rot][bits].items()))
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    bits_ok = {}
    for rot, por_bits in resC.items():
        ok = {b: all(np.isfinite(v['razao']) and v['razao'] <= 1.05 for v in por_bits[b].values()) for b in BITS}
        suficiente = None
        for b in sorted(BITS):
            if all(ok[bb] for bb in BITS if bb >= b):
                suficiente = b
                break
        bits_ok[rot] = suficiente
        print(f'  Critério C — {rot}: bits suficientes = {suficiente if suficiente else "nenhum dos testados"}'
              + ('' if rot != '74 fase + amp.' else '  (só reportado)'))

    # ═════════════════════════ figura ═════════════════════════
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True, sharey=True)
    for ax, nm in zip(axs, INST_NOMES):
        for (a, estilo) in ((D_FILTRO, ':'), (D_67, '-'), (D_74, '--')):
            ax.plot(K_GRADE, razoes_k(a, D_OTIMO, nm), estilo, marker='o', ms=3, label=f'{a} / ótimo')
        ax.axhline(1, color='#8b949e', lw=0.7)
        ax.set_xscale('log'); ax.set_title(nm); ax.set_xlabel('k (ruído / ruído real)')
        ax.grid(True, alpha=0.1)
    axs[0].set_ylabel('B·f₅₀ / B·f₅₀ do ótimo branqueado')
    axs[0].legend(fontsize=8)
    fig_path = Path(FIGURES_DIR) / 'cel75_otimo_branqueado.png'
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f'\n  Figura: {fig_path.name}   Tempo: {(time.time() - t0) / 60:.1f} min')
    save_report('75_referencias_artigo_b', {
        'fap': FAP, 'k_grade': K_GRADE, 'n_boot': N_BOOT, 'bits': list(BITS),
        'estimador': 'limiar_bf50_fisico (beta = 1); D²NN lida pela energia na janela',
        'A_bf50': resA, 'A_razoes': resumoA, 'A_criterio_atinge_otimo_crisp_k1': atinge,
        'B_fracoes': resB, 'B_transmissao_74': t74,
        'C_quantizacao': {r: {str(b): v for b, v in d.items()} for r, d in resC.items()},
        'C_bits_suficientes': bits_ok, 'figura': str(fig_path),
        'tempo_min': (time.time() - t0) / 60})


if __name__ == '__main__':
    main()
