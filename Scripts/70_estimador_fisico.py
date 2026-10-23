"""
70_estimador_fisico.py — Estimador de B·f₅₀ pela forma física da curva de detecção.

Motivação (script 69): os dois estimadores testados reprovaram no critério 1 e as razões
D²NN/filtro dependiam do estimador (mediana |iso/psi − 1| = 5,5%; critério 2 reprovado).
  • psicométrico (logística em log B·f, piso = FAP): desvio 4,3%, mas viés −11% acima de 15 G
    — a forma logística não é a forma da curva;
  • isotônico (sem forma): viés menor, mas desvio de 9% — a varredura tem poucos perfis
    acima de 15 G, e sem forma o ruído de Bernoulli passa direto para a estimativa.
Diagnóstico feito antes deste script: a referência "teoria" (z₅₀·σ_a/g) está certa. A curva
exata, média das probabilidades de detecção de cada perfil, cruza 50% a ±3% da teoria em
todos os casos; o ganho g é constante (p10–p90 = 0,9998–1,0000). O erro é do estimador.

Estimador novo (Src/evaluation.py, limiar_bf50_fisico):
  P(B·f) = Φ(u − z_c) + Φ(−u − z_c),  u = (B·f/s)^β,  z_c = Φ⁻¹(1 − FAP/2).
  É a curva exata de um detector |estatística linear| com ruído gaussiano e limiar fixado no
  falso alarme: β = 1 e um só parâmetro livre (s). O piso P(0) = FAP sai da forma.
  Teste preliminar (60 reamostragens, só o filtro): β = 1 → cobertura 88%, desvio 5,1%,
  viés −1,2% (> 15 G) e +3,7% (≤ 15 G): aprovado. β livre → viés −5,2% acima de 15 G.

Parte 1 — filtro |a| contra a teoria (varredura do 54, 200 reamostragens), psicométrico ×
  físico (β = 1). Critério 1 (o mesmo do 69): cobertura ≥ 80%, desvio mediano < 10% e
  |viés mediano| < 5% acima e abaixo de 15 G.
Parte 2 — a forma vale para a D²NN 67? A D²NN lê a ENERGIA numa janela (soma de |campo|²),
  que não é exatamente |estatística linear|. Para cada k e instrumento: ajuste com β = 1 e com
  β livre, teste da razão de verossimilhança (2ΔNLL ~ χ²₁). Filtro como controle.
  Critério 3 (definido antes): β = 1 é adequado para a D²NN se p > 0,01 em ≥ 90% dos casos
  E a mediana de |B·f₅₀(β livre)/B·f₅₀(β = 1) − 1| < 3%.
Parte 3 — razões B·f₅₀ D²NN 67/filtro com o estimador adotado (β = 1 se o critério 3 passar;
  senão β livre para os dois detectores), com IC 95% por bootstrap pareado (mesmos perfis
  reamostrados para os dois detectores), ao lado da razão psicométrica que o 54 reporta hoje.

Decisão: se o critério 1 passar, o 54 passa a usar limiar_bf50_fisico (com o β da parte 2).

Uso:
    python Scripts/70_estimador_fisico.py
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm, chi2

import torch
from torch.utils.data import TensorDataset, DataLoader

from Src.config import DEVICE, DTYPE
from Src.model import B3
from Src.physics import INSTRUMENTAL_V2
from Src.dataset import generate_degraded_profiles, INST_NOMES, SEED_BASE
from Src.train_utils import X_SCALE_TREINO, MODEL_INST_V2_PATH
from Src.evaluation import (templates_filtro_casado, amostrar_varredura_bf, limiar_bf50,
                            limiar_bf50_fisico, carregar_modelo, MontadorEntrada,
                            prever_logits, softmax)
from Src.report_utils import save_report

INSTR = INSTRUMENTAL_V2
FAP = 0.05
K_GRADE = sorted({0.1, 1 / (3.44 * np.sqrt(1.1)), 0.2, 1 / (3.11 * np.sqrt(1.1)), 0.5, 0.7, 1.0, 1.4, 2.0,
                  1 / np.sqrt(44), 1 / np.sqrt(11)})
SEED_NULO_DEG, SEED_VAR, SEED_VAR_DEG = SEED_BASE + 4003, SEED_BASE + 4002, SEED_BASE + 4004
N_BOOT = 200
Z_C = norm.isf(FAP / 2)
Z50 = brentq(lambda m: norm.sf(Z_C - m) + norm.cdf(-Z_C - m) - 0.5, 0.0, 10.0)
MODELO_67 = MODEL_INST_V2_PATH.with_name('modelo_d2nn_optico_v2.pt')


def amplitude(V, inst, templ):
    a = np.empty(len(V))
    for k in range(len(INST_NOMES)):
        m = inst == k
        t = templ[k]
        a[m] = V[m] @ t / (t @ t)
    return a


def conjunto(B, F, seed):
    _, V0, i0 = generate_degraded_profiles(B, F, seed=seed, fator_ruido=0.0, instrumental=INSTR)
    _, V1, i1 = generate_degraded_profiles(B, F, seed=seed, fator_ruido=1.0, instrumental=INSTR)
    assert (i0 == i1).all()
    return V0, V1 - V0, i0


def escores_d2nn(modelo, montar, V0, dV, k):
    X = torch.tensor(V0 + k * dV, dtype=DTYPE, device=DEVICE)
    P = softmax(prever_logits(DataLoader(TensorDataset(X), batch_size=B3['batch_size']), modelo, montar))
    return 1.0 - P[:, 0]


def deteccoes(e0, ev, i0, iv, k):
    """Detecção com falso alarme fixado em FAP pelos perfis sem campo do instrumento k."""
    thr = np.quantile(e0[i0 == k], 1 - FAP)
    return (ev[iv == k] > thr).astype(float)


# ───────────────────────────── Parte 1 ─────────────────────────────
def parte1(bf, Vv0, dVv, iv, nulo, templ):
    a_lim = amplitude(Vv0, iv, templ)
    av_ru = amplitude(dVv, iv, templ)
    V00, dV0, i0 = nulo
    a0_lim, a0_ru = amplitude(V00, i0, templ), amplitude(dV0, i0, templ)
    g = {nm: float(np.median(a_lim[(iv == k) & (bf > 0)] / bf[(iv == k) & (bf > 0)]))
         for k, nm in enumerate(INST_NOMES)}
    est = {'psicometrico': lambda b, d, r: limiar_bf50(b, d, r, n_boot=N_BOOT, piso=FAP),
           'fisico': lambda b, d, r: limiar_bf50_fisico(b, d, r, n_boot=N_BOOT, fap=FAP)}
    rng = {'psicometrico': np.random.default_rng(1001), 'fisico': np.random.default_rng(4004)}
    casos = {e: [] for e in est}
    for kf in K_GRADE:
        a0 = a0_lim + kf * a0_ru
        av = np.abs(a_lim + kf * av_ru)
        for k, nm in enumerate(INST_NOMES):
            teo = Z50 * float(np.std(a0[i0 == k])) / g[nm]
            det = deteccoes(np.abs(a0), av, i0, iv, k)
            for nome, f in est.items():
                e, lo, hi, _ = f(bf[iv == k], det, rng[nome])
                if np.isfinite(e):
                    casos[nome].append({'k': float(kf), 'instrumento': nm, 'teoria': teo, 'estimado': e,
                                        'ic': [lo, hi], 'desvio': e / teo - 1,
                                        'dentro_ic': bool(lo <= teo <= hi) if np.isfinite(lo) else False})
    print(f'\n{"="*90}\nPARTE 1 — FILTRO |a| CONTRA A TEORIA (varredura do 54: {len(bf)} perfis, '
          f'{N_BOOT} reamostragens)\n{"="*90}')
    print(f'  {"k":>6s} {"instrumento":12s} {"teoria":>7s} {"psicométrico":>20s} {"físico (β = 1)":>20s}')
    idx = {(c['k'], c['instrumento']): c for c in casos['fisico']}
    for c in casos['psicometrico']:
        cf = idx.get((c['k'], c['instrumento']))
        tf = f'{cf["estimado"]:7.1f} ({cf["desvio"]:+5.0%}){"✓" if cf["dentro_ic"] else " "}' if cf else '—'
        print(f'  {c["k"]:6.3f} {c["instrumento"]:12s} {c["teoria"]:7.1f} '
              f'{c["estimado"]:7.1f} ({c["desvio"]:+5.0%}){"✓" if c["dentro_ic"] else " "}    {tf:>20s}')
    resumo = {}
    for nome, cs in casos.items():
        d = np.array([c['desvio'] for c in cs])
        t = np.array([c['teoria'] for c in cs])
        r = {'n': len(cs), 'cobertura': float(np.mean([c['dentro_ic'] for c in cs])),
             'desvio_mediano_abs': float(np.median(np.abs(d))),
             'vies_acima_15G': float(np.median(d[t > 15])), 'vies_ate_15G': float(np.median(d[t <= 15]))}
        r['aprovado'] = bool(r['cobertura'] >= 0.80 and r['desvio_mediano_abs'] < 0.10
                             and abs(r['vies_acima_15G']) < 0.05 and abs(r['vies_ate_15G']) < 0.05)
        resumo[nome] = r
        print(f'  {nome:13s} n = {r["n"]:2d} | cobertura {100*r["cobertura"]:.0f}% | desvio mediano '
              f'{100*r["desvio_mediano_abs"]:.1f}% | viés > 15 G {100*r["vies_acima_15G"]:+.1f}% | '
              f'≤ 15 G {100*r["vies_ate_15G"]:+.1f}% → {"APROVADO" if r["aprovado"] else "REPROVADO"}')
    return {'casos': casos, 'resumo': resumo, 'ganho': g}


# ───────────────────────────── Parte 2 ─────────────────────────────
def lrt(b, d):
    """β = 1 × β livre: (B·f₅₀ β=1, B·f₅₀ β livre, β, p do teste da razão de verossimilhança)."""
    i1, il = {}, {}
    e1 = limiar_bf50_fisico(b, d, None, n_boot=0, fap=FAP, info=i1)[0]
    el = limiar_bf50_fisico(b, d, None, n_boot=0, fap=FAP, beta_livre=True, info=il)[0]
    if not (np.isfinite(e1) and np.isfinite(el)):
        return None
    stat = max(0.0, 2 * (i1['nll'] - il['nll']))
    return e1, el, il['beta'], float(chi2.sf(stat, 1))


def parte2(bf, iv, i0, escores):
    print(f'\n{"="*90}\nPARTE 2 — A FORMA COM β = 1 VALE PARA A D²NN 67? (razão de verossimilhança, '
          f'β = 1 × β livre)\n{"="*90}')
    print(f'  {"k":>6s} {"instrumento":12s} │ {"D²NN: β":>8s} {"p":>7s} {"Δ B·f₅₀":>8s} │ '
          f'{"filtro: β":>9s} {"p":>7s} {"Δ B·f₅₀":>8s}')
    res = {'d2nn': [], 'filtro': []}
    for kf in K_GRADE:
        txt = {}
        for det_nome in ('d2nn', 'filtro'):
            e0, ev = escores[det_nome][kf]
            out = []
            for k, nm in enumerate(INST_NOMES):
                r = lrt(bf[iv == k], deteccoes(e0, ev, i0, iv, k))
                if r is not None:
                    res[det_nome].append({'k': float(kf), 'instrumento': nm, 'bf50_beta1': r[0],
                                          'bf50_livre': r[1], 'beta': r[2], 'p': r[3]})
                out.append(r)
            txt[det_nome] = out
        for k, nm in enumerate(INST_NOMES):
            def f(r):
                return (f'{r[2]:8.2f} {r[3]:7.3f} {r[1]/r[0]-1:+8.1%}' if r else f'{"—":>8s} {"":7s} {"":8s}')
            print(f'  {kf:6.3f} {nm:12s} │ {f(txt["d2nn"][k])} │  {f(txt["filtro"][k])}')
    resumo = {}
    for det_nome, cs in res.items():
        p = np.array([c['p'] for c in cs])
        dlt = np.array([c['bf50_livre'] / c['bf50_beta1'] - 1 for c in cs])
        beta = np.array([c['beta'] for c in cs])
        r = {'n': len(cs), 'frac_p_maior_001': float(np.mean(p > 0.01)),
             'mediana_abs_delta': float(np.median(np.abs(dlt))), 'beta_mediano': float(np.median(beta))}
        r['beta1_adequado'] = bool(r['frac_p_maior_001'] >= 0.90 and r['mediana_abs_delta'] < 0.03)
        resumo[det_nome] = r
        print(f'  {det_nome:7s} n = {r["n"]:2d} | p > 0,01 em {100*r["frac_p_maior_001"]:.0f}% | mediana '
              f'|Δ B·f₅₀| {100*r["mediana_abs_delta"]:.1f}% | β mediano {r["beta_mediano"]:.2f} → β = 1 '
              + ('ADEQUADO' if r['beta1_adequado'] else 'INADEQUADO'))
    return {'casos': res, 'resumo': resumo}


# ───────────────────────────── Parte 3 ─────────────────────────────
def parte3(bf, iv, i0, escores, beta_livre):
    from Src.evaluation import _ajuste_fisico
    z50b = lambda ls, lb: float(np.exp(ls) * Z50 ** (1.0 / np.exp(lb)))
    rng = np.random.default_rng(5005)
    rng_psi = np.random.default_rng(3003)
    nome_est = 'físico, β livre' if beta_livre else 'físico, β = 1'
    print(f'\n{"="*90}\nPARTE 3 — RAZÃO B·f₅₀ D²NN 67 / filtro |a| ({nome_est}; IC 95% por bootstrap pareado, '
          f'{N_BOOT})\n{"="*90}')
    print(f'  {"k":>6s} {"instrumento":12s} {"filtro (G)":>10s} {"D²NN (G)":>9s} {"razão":>6s} '
          f'{"IC 95%":>15s} {"psicom. (54)":>13s}')
    linhas = []
    for kf in K_GRADE:
        for k, nm in enumerate(INST_NOMES):
            b = bf[iv == k]
            dd = deteccoes(*escores['d2nn'][kf], i0, iv, k)
            df = deteccoes(*escores['filtro'][kf], i0, iv, k)
            ed = limiar_bf50_fisico(b, dd, None, n_boot=0, fap=FAP, beta_livre=beta_livre)[0]
            ef = limiar_bf50_fisico(b, df, None, n_boot=0, fap=FAP, beta_livre=beta_livre)[0]
            psi = (limiar_bf50(b, dd, rng_psi, n_boot=0, piso=FAP)[0]
                   / limiar_bf50(b, df, rng_psi, n_boot=0, piso=FAP)[0])
            lin = {'k': float(kf), 'instrumento': nm, 'bf50_filtro': ef, 'bf50_d2nn': ed,
                   'razao': ed / ef if np.isfinite(ed) and np.isfinite(ef) else np.nan,
                   'razao_psicometrica': float(psi), 'ic': [np.nan, np.nan]}
            if np.isfinite(lin['razao']):
                rs = []
                for _ in range(N_BOOT):
                    i = rng.integers(0, len(b), len(b))
                    if dd[i].min() == dd[i].max() or df[i].min() == df[i].max():
                        continue
                    pd_ = _ajuste_fisico(b[i], dd[i], Z_C, beta_livre)
                    pf_ = _ajuste_fisico(b[i], df[i], Z_C, beta_livre)
                    rs.append(z50b(*pd_[:2]) / z50b(*pf_[:2]))
                if len(rs) >= N_BOOT // 2:
                    lin['ic'] = [float(v) for v in np.percentile(rs, [2.5, 97.5])]
            linhas.append(lin)
            if np.isfinite(lin['razao']):
                print(f'  {kf:6.3f} {nm:12s} {ef:10.1f} {ed:9.1f} {lin["razao"]:6.2f} '
                      f'  [{lin["ic"][0]:.2f}, {lin["ic"][1]:.2f}] {psi:13.2f}')
            else:
                print(f'  {kf:6.3f} {nm:12s} {"—":>10s} {"—":>9s} {"—":>6s}')
    print('\n  Por instrumento (mediana das razões em k; faixa):')
    resumo = {}
    for nm in INST_NOMES:
        r = np.array([l['razao'] for l in linhas if l['instrumento'] == nm and np.isfinite(l['razao'])])
        ps = np.array([l['razao_psicometrica'] for l in linhas
                       if l['instrumento'] == nm and np.isfinite(l['razao_psicometrica'])])
        lim1 = [l for l in linhas if l['instrumento'] == nm and abs(l['k'] - 1.0) < 1e-9]
        resumo[nm] = {'mediana': float(np.median(r)), 'min': float(r.min()), 'max': float(r.max()),
                      'mediana_psicometrica': float(np.median(ps)),
                      'k1': lim1[0] if lim1 else None}
        txt_k1 = (f' | k = 1: {lim1[0]["razao"]:.2f} [{lim1[0]["ic"][0]:.2f}, {lim1[0]["ic"][1]:.2f}]'
                  if lim1 and np.isfinite(lim1[0]['razao']) else '')
        print(f'  {nm:12s} {np.median(r):.2f} ({r.min():.2f}–{r.max():.2f}) | psicométrico '
              f'{np.median(ps):.2f}{txt_k1}')
    return {'estimador': nome_est, 'linhas': linhas, 'por_instrumento': resumo}


def main():
    t0 = time.time()
    templ = templates_filtro_casado(INST_NOMES, instrumental=INSTR)
    B0 = np.zeros(1500)
    nulo = conjunto(B0, B0, SEED_NULO_DEG)
    V00, dV0, i0 = nulo
    print(f'z_c = {Z_C:.3f} | z₅₀ = {Z50:.3f} | k: ' + ', '.join(f'{k:.3f}' for k in K_GRADE))
    Bv, Fv, _ = amostrar_varredura_bf(1000, SEED_VAR, classes=(0, 1))
    bf = Bv * Fv
    Vv0, dVv, iv = conjunto(Bv, Fv, SEED_VAR_DEG)

    r1 = parte1(bf, Vv0, dVv, iv, nulo, templ)
    ok1 = r1['resumo']['fisico']['aprovado']

    r2 = r3 = None
    if MODELO_67.exists():
        modelo, meta = carregar_modelo(MODELO_67, DEVICE)
        montar = MontadorEntrada(float(meta.get('X_scale', X_SCALE_TREINO)), DEVICE)
        escores = {'d2nn': {}, 'filtro': {}}
        for kf in K_GRADE:
            escores['d2nn'][kf] = (escores_d2nn(modelo, montar, V00, dV0, kf),
                                   escores_d2nn(modelo, montar, Vv0, dVv, kf))
            escores['filtro'][kf] = (np.abs(amplitude(V00 + kf * dV0, i0, templ)),
                                     np.abs(amplitude(Vv0 + kf * dVv, iv, templ)))
        r2 = parte2(bf, iv, i0, escores)
        beta_livre = not r2['resumo']['d2nn']['beta1_adequado']
        r3 = parte3(bf, iv, i0, escores, beta_livre)
    else:
        print(f'\n  ⚠ {MODELO_67.name} não encontrado — partes 2 e 3 puladas')

    print(f'\n{"="*90}\nDECISÃO\n{"="*90}')
    print(f'  Critério 1 (físico, filtro): {"APROVADO" if ok1 else "REPROVADO"}')
    if r2:
        ok3 = r2['resumo']['d2nn']['beta1_adequado']
        print(f'  Critério 3 (β = 1 na D²NN): {"APROVADO" if ok3 else "REPROVADO → β livre nos dois detectores"}')
    if ok1:
        print('  → o 54 passa a usar limiar_bf50_fisico' + ('' if not r2 else
              (' com β = 1' if r2['resumo']['d2nn']['beta1_adequado'] else ' com β livre')))
    else:
        print('  → manter o 54 como está; o estimador precisa de revisão')
    print(f'  Tempo: {(time.time() - t0) / 60:.1f} min')
    save_report('70_estimador_fisico', {'z_c': Z_C, 'z50': Z50, 'fap': FAP, 'k_grade': K_GRADE,
                                        'n_boot': N_BOOT, 'parte1': r1, 'criterio1': ok1,
                                        'parte2': r2, 'parte3': r3})


if __name__ == '__main__':
    main()
