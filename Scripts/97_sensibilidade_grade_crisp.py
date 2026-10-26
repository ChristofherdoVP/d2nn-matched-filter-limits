"""
97_sensibilidade_grade_crisp.py — artigo B: quanto os limites de detecção do CRISP dependem da grade λ?

O artigo B modela o CRISP com uma grade REPRESENTATIVA de 15 pontos em ±90 pm (Src/physics.py,
GRADE_CRISP_PM_PROVISORIA; texto: "typical 15-point observing programmes"), porque a exportação de
2024-07-08 não tinha WCS-TAB. As observações CRISP de 2024-10-08 (arquivo do SST; grade lida da tabela
WCS-TAB desses arquivos, valores fixos abaixo — não é preciso baixar dados para rodar) usam
−650 … +700 mÅ em relação a 8542,09 Å, isto é, ±67,5 pm em torno do próprio centro (+25 mÅ),
com passo de 75 mÅ no centro e 150 mÅ nas asas. O ruído do B (σ_k = 3,2×10⁻³ I_k) foi confirmado
nesses dados na análise observacional do artigo companheiro (não incluída neste repositório).
Suplemento 1, Seção S3.

Os dois detectores de referência do B são lineares em V, então a₅₀ = z₅₀ · s (Apêndice A do B), com
s o desvio de â só de ruído. A razão a₅₀(grade nova)/a₅₀(grade do B) = s_nova/s_B, calculada
analiticamente com a mesma cadeia do B (perfil de Sol calmo e template por G de B·f, degradados sem
ruído por degradar_stokes; σ_k = ε I_k):
  filtro casado BRANQUEADO na grade nativa:  s_w = (Σ t_k²/σ_k²)^(−1/2)
  filtro casado SEM PESOS na grade de processamento (LAM, 3000 pontos; V = 0 fora da faixa):
      g = t_LAM/(t_LAM · t_LAM),  pesos nativos w = Mᵀ g (M = interpolação nativa → LAM),
      s_u = √(Σ w_k² σ_k²)
Grades testadas (pm em relação ao núcleo do modelo, λ₀_core):
  B        : grade provisória do B (±90 pm)
  out2024  : grade de 2024-10-08 centrada no núcleo (±67,5 pm)               ← decisão
  out2024+ : a mesma com o deslocamento real de +2,5 pm entre o centro da grade e 8542,09 Å
             (descritivo; a posição do núcleo observado em relação a 8542,09 Å depende da velocidade)

Critério (definido antes de rodar), na grade out2024, para os DOIS detectores:
  |a₅₀(out2024)/a₅₀(B) − 1| ≤ 5%   → GRADE IRRELEVANTE: o B não muda (frase opcional do autor);
  5% < |…| ≤ 15%                    → NOTA DE SENSIBILIDADE no suplemento, sem retreinar;
  > 15%                             → avaliar as máscaras treinadas na grade nova (e talvez retreinar).
Descritivo: a₅₀ projetados = valores da Tabela 2 do B × razão (CRISP: branqueado 47,5 G; sem pesos 57,8 G).

Uso:  python Scripts/97_sensibilidade_grade_crisp.py
Saída: Outputs/Reports/97_sensibilidade_grade_crisp.json, Outputs/Figures/cel97_grade_crisp.png
"""
import sys
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Src.physics import INSTRUMENTAL_V2, LAM, V_CORE_KMS, get_stokes, degradar_stokes

REP = ROOT / 'Outputs' / 'Reports'
FIG = ROOT / 'Outputs' / 'Figures'
B_TAB2 = {'branqueado': 47.5, 'sem_pesos': 57.8}           # CRISP, Tabela 2 do artigo B (G)
OUT_MA = np.array([-650, -500, -350, -275, -200, -125, -50, 25, 100, 175, 250, 325, 400, 550, 700], float)
GRADES = {
    'B': np.asarray(INSTRUMENTAL_V2['CRISP/SST']['grade_dl_pm'], dtype=float),
    'out2024': (OUT_MA - 25.0) / 10.0,
    'out2024+': OUT_MA / 10.0 - 0.0,      # centro da grade 2,5 pm ao vermelho do núcleo
}
GRADES['out2024+'] = (OUT_MA - 25.0) / 10.0 + 2.5


def s_detectores(grade_pm):
    inst = copy.deepcopy(INSTRUMENTAL_V2['CRISP/SST'])
    inst['grade_dl_pm'] = grade_pm
    eps = float(inst['ruido']['eps'])
    rng = np.random.default_rng(0)
    d0 = degradar_stokes(get_stokes(0.0, filling=1.0, v_km_s=V_CORE_KMS), LAM, inst, rng_local=rng, fator_ruido=0.0)
    d1 = degradar_stokes(get_stokes(100.0, filling=1.0, v_km_s=V_CORE_KMS), LAM, inst, rng_local=rng, fator_ruido=0.0)
    I = d0['I'].astype(float)
    t = d1['V'].astype(float) / 100.0                 # V por G de B·f na grade nativa
    lam_obs = d1['lam_obs']
    sig = eps * np.abs(I)
    s_w = 1.0 / np.sqrt(np.sum(t ** 2 / sig ** 2))
    n = len(lam_obs)
    M = np.column_stack([np.interp(LAM, lam_obs, np.eye(n)[k], left=0.0, right=0.0) for k in range(n)])
    tL = M @ t
    g = tL / (tL @ tL)
    w = M.T @ g
    s_u = float(np.sqrt(np.sum(w ** 2 * sig ** 2)))
    return {'s_branqueado': float(s_w), 's_sem_pesos': s_u, 'I': I.tolist(), 't': t.tolist(),
            'lam_pm': ((lam_obs - lam_obs[np.argmin(np.abs(np.asarray(grade_pm)))]) * 1e3).tolist(),
            'grade_pm': list(map(float, grade_pm))}


def main():
    res = {k: s_detectores(g) for k, g in GRADES.items()}
    base = res['B']
    out = {'grades_pm': {k: list(map(float, g)) for k, g in GRADES.items()}, 'resultados': {}}
    print('grade       s branqueado   s sem pesos   razão branq.   razão sem pesos   a₅₀ projetado (branq. / sem pesos)')
    for k, r in res.items():
        rw = r['s_branqueado'] / base['s_branqueado']
        ru = r['s_sem_pesos'] / base['s_sem_pesos']
        out['resultados'][k] = {'s_branqueado': r['s_branqueado'], 's_sem_pesos': r['s_sem_pesos'],
                                'razao_branqueado': rw, 'razao_sem_pesos': ru,
                                'a50_projetado_G': {'branqueado': B_TAB2['branqueado'] * rw,
                                                    'sem_pesos': B_TAB2['sem_pesos'] * ru}}
        print(f"{k:10s}  {r['s_branqueado']:10.3f}    {r['s_sem_pesos']:10.3f}     {rw:8.3f}        {ru:8.3f}"
              f"          {B_TAB2['branqueado'] * rw:5.1f} G / {B_TAB2['sem_pesos'] * ru:5.1f} G")
    rw = out['resultados']['out2024']['razao_branqueado']
    ru = out['resultados']['out2024']['razao_sem_pesos']
    dev = max(abs(rw - 1), abs(ru - 1))
    if dev <= 0.05:
        dec = 'GRADE IRRELEVANTE (B não muda)'
    elif dev <= 0.15:
        dec = 'NOTA DE SENSIBILIDADE NO SUPLEMENTO (sem retreinar)'
    else:
        dec = 'AVALIAR MÁSCARAS NA GRADE NOVA'
    out['desvio_maximo_out2024'] = dev
    out['decisao'] = dec
    print(f"\nDesvio máximo na grade out2024: {dev * 100:.1f}%  →  {dec}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    cores = {'B': '#999999', 'out2024': '#0072B2', 'out2024+': '#D55E00'}
    for k, r in res.items():
        g = np.asarray(r['grade_pm'])
        axs[0].plot(g, r['I'], 'o-', color=cores[k], ms=4, label=k)
        axs[1].plot(g, np.asarray(r['t']) * 100, 'o-', color=cores[k], ms=4, label=k)
    axs[0].set_xlabel('Δλ (pm, em relação ao núcleo do modelo)'); axs[0].set_ylabel('I degradado (Ic)')
    axs[1].set_xlabel('Δλ (pm)'); axs[1].set_ylabel('V para B·f = 100 G (Ic)')
    axs[0].legend(fontsize=8)
    fig.suptitle(f'CRISP: a₅₀ grade 2024-10 / grade do B = {rw:.3f} (branqueado), {ru:.3f} (sem pesos)  →  {dec}',
                 fontsize=9)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / 'cel97_grade_crisp.png', dpi=120)
    plt.close(fig)
    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / '97_sensibilidade_grade_crisp.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Relatório: {REP / '97_sensibilidade_grade_crisp.json'}\nFigura: {FIG / 'cel97_grade_crisp.png'}")


if __name__ == '__main__':
    main()
