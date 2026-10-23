"""
68_doppler.py — Robustez ao deslocamento Doppler: D²NN óptica × filtro casado.

Substitui as Células 31 (Doppler v ∈ [−5, 5] km/s) e parte da 34 do Colab.

Física:
  - O filtro casado tem template FIXO (velocidade do treino): um desvio Doppler descasa o
    template e o filtro perde sinal. Solução clássica: banco de templates em várias
    velocidades, com o máximo — paga um limiar mais alto (mais chances de o ruído passar).
  - O 4f é equivariante a translações em λ: na D²NN óptica o desvio Doppler só DESLOCA o
    pico do POF (≈ 8,5 pixels por km/s na grade LAM). Com um detector só (janela 3 × 5), o pico
    sai da janela. Com uma FILEIRA de detectores ao longo de λ e um comparador no máximo, a
    óptica é a mesma e só a leitura cresce.

Desvios testados (em relação ao núcleo do treino): Δv = 0; 0,5; 1; 2; 3; 5 e −2 km/s.
Para cada Δv, perfis v2 com ruído real (k = 1), mesmos B, f e sementes do 54:
  - conjunto natural (500/classe): acurácia por argmax;
  - sem campo (1500) e varredura Quieta + Rede (1000/classe): B·f₅₀ a 5% de falso alarme por
    instrumento (limiar = quantil de 95% do escore nos perfis sem campo, no mesmo Δv).

Classificadores:
  67           : D²NN óptica (um detector)
  67 + fileira : mesma óptica; escore = máximo da energia em janelas 3 × 5 deslocadas de −45 a
                 +45 pixels (19 detectores, passo 5 px ≈ 0,6 km/s; cobre ±5 km/s)
  filtro |a|   : template na velocidade do treino
  banco |a|    : templates de −6 a +6 km/s (passo 0,5), escore = máximo |a|
A acurácia por argmax da fileira usa os limiares do 67 (treinados com um detector): serve só
como indicação; a comparação justa é o B·f₅₀ a falso alarme fixo.

Critério (definido antes):
  CD  67 + fileira com B·f₅₀ ≤ 1,1 × o do banco de filtros para |Δv| ≤ 3 km/s, no IBIS e no
      CRISP (a D²NN óptica é tão robusta ao Doppler quanto o banco).
Descritivo: degradação relativa B·f₅₀(Δv)/B·f₅₀(0) de cada método.

Uso:
    python Scripts/68_doppler.py
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from Src.config import DEVICE, DTYPE, FIGURES_DIR, SEED
from Src.model import B3
from Src.physics import INSTRUMENTAL_V2, V_CORE_KMS, CALIB, DLAM
from Src.dataset import generate_master_dataset, generate_degraded_profiles, INST_NOMES, NOISE_STD, SEED_BASE
from Src.train_utils import X_SCALE_TREINO, MODEL_INST_V2_PATH, split_indices
from Src.evaluation import (carregar_modelo, MontadorEntrada, softmax, prever_logits, resumo_classificacao,
                            templates_filtro_casado, ajustar_limiares, classificar_filtro_casado,
                            amostrar_varredura_bf, limiar_bf50)
from Src.report_utils import save_report

INSTR = INSTRUMENTAL_V2
DV = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, -2.0)
BANCO_V = np.arange(-6.0, 6.0 + 1e-9, 0.5)
OFFSETS_PX = np.arange(-45, 46, 5)
FAP = 0.05
SEED_NAT, SEED_NAT_DEG = SEED_BASE + 4000, SEED_BASE + 4001          # como no 54
SEED_NULO_DEG, SEED_VAR, SEED_VAR_DEG = SEED_BASE + 4003, SEED_BASE + 4002, SEED_BASE + 4004
NOMES = B3['class_names']
MODELO_67 = MODEL_INST_V2_PATH.with_name('modelo_d2nn_optico_v2.pt')
PX_POR_KMS = CALIB['lambda0'] / CALIB['c_light'] / DLAM


def loader(V):
    return DataLoader(TensorDataset(torch.tensor(np.asarray(V), dtype=DTYPE, device=DEVICE)),
                      batch_size=B3['batch_size'])


def energias_fileira(modelo, montar, V):
    """(N, n_offsets) energias em janelas iguais à do modelo, deslocadas ao longo de λ."""
    hy, hx = modelo.janela[0] // 2, modelo.janela[1] // 2
    out = []
    modelo.eval()
    with torch.no_grad():
        for (Vb,) in loader(V):
            x = montar(Vb)
            if modelo.diafragma is not None:
                x = x * modelo.diafragma
            I = modelo.mascara(x, aplicar_pad=False).abs() ** 2
            ny, nx = I.shape[-2], I.shape[-1]
            I = torch.roll(I, shifts=(ny // 2, nx // 2), dims=(-2, -1))
            cy, cx = ny // 2, nx // 2
            col = [I[..., cy - hy: cy + hy + 1, cx + o - hx: cx + o + hx + 1].sum(dim=(-2, -1)) for o in OFFSETS_PX]
            out.append(torch.stack(col, dim=1).cpu().numpy())
    return np.concatenate(out)


def amplitudes(V, inst, templ_por_v):
    """|a| para cada template: (N, n_templates). templ_por_v: lista de {inst: template}."""
    A = np.empty((len(V), len(templ_por_v)))
    for k in range(len(INST_NOMES)):
        m = inst == k
        if not m.any():
            continue
        T = np.stack([t[k] for t in templ_por_v])                 # (n_t, N_LAM)
        A[m] = np.abs(V[m] @ T.T / np.sum(T * T, axis=1))
    return A


def b50(esc0, i0, escv, iv, bf, rng):
    out = {}
    for k, nm in enumerate(INST_NOMES):
        thr = np.quantile(esc0[i0 == k], 1 - FAP)
        det = (escv[iv == k] > thr).astype(float)
        r = limiar_bf50(bf[iv == k], det, rng, n_boot=0, piso=FAP)
        out[nm] = r[0] if np.isfinite(r[0]) else (np.inf if 'acima' in r[3] else np.nan)
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    modelo, meta = carregar_modelo(MODELO_67, DEVICE)
    montar = MontadorEntrada(float(meta.get('X_scale', X_SCALE_TREINO)), DEVICE)
    print(f'Modelo: {MODELO_67.name} | janela {modelo.janela} | diafragma ±{modelo.diafragma_pm} pm | '
          f'{PX_POR_KMS:.2f} px por km/s | fileira: {len(OFFSETS_PX)} detectores de {OFFSETS_PX[0]} a '
          f'{OFFSETS_PX[-1]} px (±{OFFSETS_PX[-1] / PX_POR_KMS:.1f} km/s)')

    templ0 = [templates_filtro_casado(INST_NOMES, v_km_s=V_CORE_KMS, instrumental=INSTR)]
    banco = [templates_filtro_casado(INST_NOMES, v_km_s=V_CORE_KMS + v, instrumental=INSTR) for v in BANCO_V]
    # limiares de classificação dos filtros: treino em Δv = 0 (como no 57/62)
    (_, y_tr, B_tr, F_tr, *_) = generate_master_dataset(NOISE_STD, SEED_BASE, 'static')
    _, V_tr, i_tr = generate_degraded_profiles(B_tr, F_tr, seed=SEED + 2, instrumental=INSTR)
    idx_train, _, _ = split_indices(y_tr, B3, SEED)
    lim_fc = ajustar_limiares(amplitudes(V_tr[idx_train], i_tr[idx_train], templ0)[:, 0], y_tr[idx_train])
    lim_bc = ajustar_limiares(amplitudes(V_tr[idx_train], i_tr[idx_train], banco).max(1), y_tr[idx_train])

    (_, y_n, B_n, F_n, *_) = generate_master_dataset(NOISE_STD, SEED_NAT, 'static', n_samples=500)
    B0 = np.zeros(1500)
    Bv, Fv, _ = amostrar_varredura_bf(1000, SEED_VAR, classes=(0, 1))
    bf = Bv * Fv
    metodos = ('67', '67 + fileira', 'filtro |a|', 'banco |a|')
    res = {}
    for dv in DV:
        v = V_CORE_KMS + dv
        _, Vn, i_n = generate_degraded_profiles(B_n, F_n, seed=SEED_NAT_DEG, v_km_s=v, instrumental=INSTR)
        _, V0, i0 = generate_degraded_profiles(B0, B0, seed=SEED_NULO_DEG, v_km_s=v, instrumental=INSTR)
        _, Vv, iv = generate_degraded_profiles(Bv, Fv, seed=SEED_VAR_DEG, v_km_s=v, instrumental=INSTR)

        # escores e previsões
        P = {nome: softmax(prever_logits(loader(X), modelo, montar))
             for nome, X in (('n', Vn), ('0', V0), ('v', Vv))}
        Ef = {nome: energias_fileira(modelo, montar, X) for nome, X in (('n', Vn), ('0', V0), ('v', Vv))}
        Afc = {nome: amplitudes(X, ii, templ0)[:, 0] for nome, X, ii in (('n', Vn, i_n), ('0', V0, i0), ('v', Vv, iv))}
        Abc = {nome: amplitudes(X, ii, banco).max(1) for nome, X, ii in (('n', Vn, i_n), ('0', V0, i0), ('v', Vv, iv))}

        def logits_fileira(E):
            z = (np.log(E.max(1) + 1e-12) - float(modelo.mu)) / float(modelo.s)
            w, b = modelo.w.detach().cpu().numpy(), modelo.b.detach().cpu().numpy()
            return w[None, :] * z[:, None] + b[None, :]

        pred = {'67': P['n'].argmax(1), '67 + fileira': logits_fileira(Ef['n']).argmax(1),
                'filtro |a|': classificar_filtro_casado(Afc['n'], lim_fc),
                'banco |a|': classificar_filtro_casado(Abc['n'], lim_bc)}
        esc = {'67': (1 - P['0'][:, 0], 1 - P['v'][:, 0]),
               '67 + fileira': (np.log(Ef['0'].max(1)), np.log(Ef['v'].max(1))),
               'filtro |a|': (Afc['0'], Afc['v']), 'banco |a|': (Abc['0'], Abc['v'])}
        res[dv] = {}
        for mt in metodos:
            r = resumo_classificacao(y_n, pred[mt], i_n, NOMES, INST_NOMES, '', verbose=False)
            res[dv][mt] = {'acuracia': r['acuracia'], 'rede': r['por_classe'][1], 'mancha': r['por_classe'][2],
                           'bf50': b50(esc[mt][0], i0, esc[mt][1], iv, bf, rng)}
        print(f'  Δv = {dv:+.1f} km/s: concluído ({time.time() - t0:.0f} s)')

    # --- Tabelas ---
    print(f'\n{"="*100}\nACURÁCIA (argmax; a da fileira usa os limiares de um detector — só indicativa)\n{"="*100}')
    print(f'  {"Δv [km/s]":>9s}' + ''.join(f'{m:>18s}' for m in metodos))
    for dv in DV:
        print(f'  {dv:>+9.1f}' + ''.join(f'{res[dv][m]["acuracia"]:>10.1f}% R{res[dv][m]["rede"]:>4.0f}%' for m in metodos))
    for nm in INST_NOMES:
        print(f'\n  B·f₅₀ a 5% FA [G] — {nm}   (entre parênteses: relativo a Δv = 0 do próprio método)')
        print(f'  {"Δv [km/s]":>9s}' + ''.join(f'{m:>18s}' for m in metodos))
        for dv in DV:
            cel = []
            for m in metodos:
                x, x0 = res[dv][m]['bf50'][nm], res[0.0][m]['bf50'][nm]
                cel.append(f'{x:>9.1f} ({x / x0:4.2f})' if np.isfinite(x) and np.isfinite(x0) and x0 > 0
                           else f'{"> faixa" if x == np.inf else "—":>16s}')
            print(f'  {dv:>+9.1f}' + ''.join(f'{c:>18s}' for c in cel))

    ok = []
    for dv in DV:
        if abs(dv) <= 3.0:
            for nm in ('IBIS/DST', 'CRISP/SST'):
                a, b = res[dv]['67 + fileira']['bf50'][nm], res[dv]['banco |a|']['bf50'][nm]
                ok.append(np.isfinite(a) and np.isfinite(b) and a <= 1.1 * b)
    cd = bool(all(ok))
    print(f'\n  CD (67 + fileira ≤ 1,1 × banco, |Δv| ≤ 3 km/s, IBIS e CRISP): {"ATENDIDO" if cd else "NÃO ATENDIDO"} '
          f'({sum(ok)}/{len(ok)} casos)')

    # --- Figura ---
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    fig.suptitle('Doppler: B·f₅₀ a 5% de falso alarme em função do desvio de velocidade', fontsize=12,
                 fontweight='bold')
    dvs = sorted(DV)
    cores = {'67': '#f85149', '67 + fileira': '#3fb950', 'filtro |a|': '#8b949e', 'banco |a|': '#58a6ff'}
    for a, nm in zip(axs, INST_NOMES):
        for m in metodos:
            ys = [res[d][m]['bf50'][nm] for d in dvs]
            a.plot(dvs, [y if np.isfinite(y) else np.nan for y in ys], 'o-', color=cores[m], label=m)
        a.set_yscale('log'); a.set_title(nm); a.set_xlabel('Δv [km/s]'); a.grid(True, alpha=0.08)
    axs[0].set_ylabel('B·f₅₀ [G]'); axs[0].legend(fontsize=8)
    out = FIGURES_DIR / 'cel68_doppler.png'
    fig.savefig(out, dpi=180, bbox_inches='tight'); plt.close(fig)
    print(f'\n  Figura: {out.name}   Tempo: {(time.time() - t0) / 60:.1f} min')
    save_report('68_doppler', {
        'dv_kms': list(DV), 'banco_v_kms': BANCO_V.tolist(), 'offsets_px': OFFSETS_PX.tolist(),
        'px_por_kms': PX_POR_KMS, 'fap': FAP, 'modelo': str(MODELO_67),
        'resultados': {str(dv): {m: {**v, 'bf50': {k: (None if not np.isfinite(x) else float(x))
                                                   for k, x in v['bf50'].items()}}
                                 for m, v in r.items()} for dv, r in res.items()},
        'criterio_CD': cd, 'tempo_min': (time.time() - t0) / 60})


if __name__ == '__main__':
    main()
