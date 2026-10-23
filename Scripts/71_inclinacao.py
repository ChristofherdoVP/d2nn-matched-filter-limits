"""
71_inclinacao.py — Inclinação do campo θ: D²NN óptica 67 × filtro casado.

Substitui as Células 33 e 33B do Colab (que usavam cos²θ — incorreto para V).

Física: no campo fraco V ∝ B·f·cosθ (componente do campo na linha de visada). I depende de θ
só pelo peso das componentes σ e π (efeito de segunda ordem em B), e no CRISP o ruído é
proporcional a I. Previsão: o limite de detecção em B·f·cosθ (= B_los·f) não depende de θ,
isto é, B·f₅₀(θ) = B·f₅₀(0)/cosθ, para os dois detectores.

Isto é uma VERIFICAÇÃO DE CONSISTÊNCIA, não uma descoberta (teste no ambiente de
desenvolvimento, só com o filtro: razões = 1,000 em todos os θ e instrumentos): o motor físico tem V ∝ cosθ por
construção. O que o script testa é se alguma etapa da cadeia (I dependente de θ → ruído
proporcional a I no CRISP; D²NN treinada só com θ = 0) quebra a escala. Para o artigo B, a
figura sustenta a frase "os limites valem para B_los·f".

Montagem: mesmos f e sementes da varredura do 54 (Quieta + Rede), ruído real (k = 1),
θ = 0°, 30°, 45°, 60° e 75° (χ = 0). Para cada θ o campo é B/cosθ, de modo que B_los·f é o
MESMO da varredura do 54 (0,05–150 G) e a mesma realização de ruído cai no mesmo perfil.
Assim a comparação entre θ não sofre com faixa nem com amostragem: só a física muda (forma de
I com θ, ruído ∝ I no CRISP, campo total até 1/cos75° ≈ 3,9× maior — ainda campo fraco para
Ca II 854,2 nm). O limiar (5% de falso alarme) vem dos perfis sem campo do 54 (θ não importa
com B = 0). B_los·f₅₀ estimado com limiar_bf50_fisico (β = 1; script 70), IC 95% por bootstrap.
(Uma primeira montagem, com o mesmo B para todo θ, comprimia a faixa de B_los·f — até 39 G em
75° — e o estimador ficava na borda: descartada antes de rodar.)

Critério (definido antes): B_los·f₅₀(θ)/B_los·f₅₀(0) dentro de 1 ± 0,05 para todo θ ≤ 75°,
nos três instrumentos, para a D²NN e para o filtro.

Uso:
    python Scripts/71_inclinacao.py
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

from Src.config import DEVICE, DTYPE, FIGURES_DIR
from Src.model import B3
from Src.physics import INSTRUMENTAL_V2, V_CORE_KMS, LAM, get_stokes, degradar_stokes, reamostrar_para_LAM
from Src.dataset import generate_degraded_profiles, INST_NOMES, INST_PROBS, SEED_BASE
from Src.train_utils import X_SCALE_TREINO, MODEL_INST_V2_PATH
from Src.evaluation import (templates_filtro_casado, amostrar_varredura_bf, limiar_bf50_fisico,
                            carregar_modelo, MontadorEntrada, prever_logits, softmax)
from Src.report_utils import save_report

INSTR = INSTRUMENTAL_V2
FAP = 0.05
THETAS = [0.0, 30.0, 45.0, 60.0, 75.0]
SEED_NULO_DEG, SEED_VAR, SEED_VAR_DEG = SEED_BASE + 4003, SEED_BASE + 4002, SEED_BASE + 4004
N_BOOT = 200
MODELO_67 = MODEL_INST_V2_PATH.with_name('modelo_d2nn_optico_v2.pt')


def perfis_theta(B, F, theta, seed):
    """Como generate_degraded_profiles (mesma sequência aleatória), com inclinação θ e campo
    total B/cosθ (B_los = B)."""
    rng = np.random.default_rng(seed)
    V = np.empty((len(B), len(LAM)))
    inst = np.empty(len(B), dtype=np.int64)
    for i in range(len(B)):
        k = int(rng.choice(len(INST_NOMES), p=INST_PROBS))
        inst[i] = k
        st = get_stokes(float(B[i]) / np.cos(np.radians(theta)), theta_deg=theta, filling=float(F[i]), v_km_s=V_CORE_KMS)
        deg = degradar_stokes(st, LAM, INSTR[INST_NOMES[k]], rng_local=rng, fator_ruido=1.0)
        V[i] = reamostrar_para_LAM(deg, INSTR[INST_NOMES[k]])[1]
    return V, inst


def amplitude(V, inst, templ):
    a = np.empty(len(V))
    for k in range(len(INST_NOMES)):
        m = inst == k
        t = templ[k]
        a[m] = V[m] @ t / (t @ t)
    return a


def escores_d2nn(modelo, montar, V):
    X = torch.tensor(V, dtype=DTYPE, device=DEVICE)
    P = softmax(prever_logits(DataLoader(TensorDataset(X), batch_size=B3['batch_size']), modelo, montar))
    return 1.0 - P[:, 0]


def main():
    t0 = time.time()
    if not MODELO_67.exists():
        sys.exit(f'{MODELO_67} não encontrado (rode o script 67).')
    modelo, meta = carregar_modelo(MODELO_67, DEVICE)
    montar = MontadorEntrada(float(meta.get('X_scale', X_SCALE_TREINO)), DEVICE)
    templ = templates_filtro_casado(INST_NOMES, instrumental=INSTR)

    B0 = np.zeros(1500)
    _, V_nulo, i0 = generate_degraded_profiles(B0, B0, seed=SEED_NULO_DEG, fator_ruido=1.0, instrumental=INSTR)
    nulo = {'d2nn': escores_d2nn(modelo, montar, V_nulo), 'filtro': np.abs(amplitude(V_nulo, i0, templ))}
    thr = {d: {k: float(np.quantile(e[i0 == k], 1 - FAP)) for k in range(len(INST_NOMES))}
           for d, e in nulo.items()}

    Bv, Fv, _ = amostrar_varredura_bf(1000, SEED_VAR, classes=(0, 1))
    bf = Bv * Fv
    rng = np.random.default_rng(7107)
    res = {}
    for th in THETAS:
        V, iv = perfis_theta(Bv, Fv, th, SEED_VAR_DEG)
        esc = {'d2nn': escores_d2nn(modelo, montar, V), 'filtro': np.abs(amplitude(V, iv, templ))}
        bf_los = bf                                   # B_los·f, igual para todo θ
        res[th] = {}
        for det, e in esc.items():
            res[th][det] = {}
            for k, nm in enumerate(INST_NOMES):
                m = iv == k
                d = (e[m] > thr[det][k]).astype(float)
                est, lo, hi, motivo = limiar_bf50_fisico(bf_los[m], d, rng, n_boot=N_BOOT, fap=FAP)
                res[th][det][nm] = {'bf50_los': est, 'ic': [lo, hi], 'motivo': motivo,
                                    'bf50_total': est / np.cos(np.radians(th)) if np.isfinite(est) else np.nan}
        print(f'  θ = {th:4.0f}°: concluído ({time.time() - t0:.0f} s)')

    print(f'\n{"="*96}\nB_los·f₅₀ a 5% de falso alarme [G] (k = 1; campo total B_los/cosθ); entre parênteses: relativo a θ = 0\n{"="*96}')
    print(f'  {"θ":>5s} │ ' + ' │ '.join(f'{nm:^25s}' for nm in INST_NOMES))
    print(f'  {"":5s} │ ' + ' │ '.join(f'{"D²NN 67":>12s} {"filtro":>12s}' for _ in INST_NOMES))
    linhas, ok = [], True
    for th in THETAS:
        txt = []
        for nm in INST_NOMES:
            cel = []
            for det in ('d2nn', 'filtro'):
                v = res[th][det][nm]['bf50_los']
                v0 = res[0.0][det][nm]['bf50_los']
                r = v / v0 if np.isfinite(v) and np.isfinite(v0) else np.nan
                res[th][det][nm]['relativo'] = r
                if th > 0 and not (np.isfinite(r) and abs(r - 1) <= 0.05):
                    ok = False
                cel.append(f'{v:5.1f} ({r:5.3f})' if np.isfinite(r) else f'{"—":>13s}')
            txt.append(f'{cel[0]:>13s} {cel[1]:>13s}')
        print(f'  {th:4.0f}° │ ' + ' │ '.join(txt))
    print(f'\n  Critério (θ ≤ 75°, todos os instrumentos, D²NN e filtro, |relativo − 1| ≤ 0,05): '
          + ('ATENDIDO — o limite vale para B_los·f' if ok else 'NÃO ATENDIDO'))

    # Figura: B·f₅₀ total contra θ, com a previsão B·f₅₀(0)/cosθ
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    tt = np.linspace(0, 78, 100)
    for ax, nm in zip(axs, INST_NOMES):
        for det, cor, rot in (('d2nn', '#2f81f7', 'D²NN 67'), ('filtro', '#d29922', 'filtro |a|')):
            y = np.array([res[th][det][nm]['bf50_total'] for th in THETAS])
            ax.plot(THETAS, y, 'o', color=cor, label=rot)
            y0 = res[0.0][det][nm]['bf50_los']
            if np.isfinite(y0):
                ax.plot(tt, y0 / np.cos(np.radians(tt)), color=cor, lw=1, ls=':')
        ax.set_yscale('log')
        ax.set_xlabel('θ [°]')
        ax.set_title(nm)
    axs[0].set_ylabel('B·f₅₀ [G] (5% FA, k = 1)')
    axs[0].legend(title='pontilhado: B·f₅₀(0)/cosθ', fontsize=8, title_fontsize=8)
    fig_path = Path(FIGURES_DIR) / 'cel71_inclinacao.png'
    fig.savefig(fig_path, dpi=150)
    print(f'\n  Figura: {fig_path.name}   Tempo: {(time.time() - t0) / 60:.1f} min')

    save_report('71_inclinacao', {'thetas': THETAS, 'fap': FAP, 'k': 1.0, 'n_boot': N_BOOT,
                                  'estimador': 'limiar_bf50_fisico (beta = 1)', 'criterio_atendido': ok,
                                  'resultados': {str(th): r for th, r in res.items()}})


if __name__ == '__main__':
    main()
