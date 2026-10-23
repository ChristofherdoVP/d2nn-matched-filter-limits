"""
67_d2nn_optica_diafragma.py — D²NN óptica com DIAFRAGMA DE CAMPO (janela espectral).

Motivação (54 com o 62): a D²NN puramente óptica atinge o filtro casado no IBIS
(B·f₅₀ 1,00–1,09×) e o supera no CRISP (0,84–0,97×, ponderação do ruído ∝ I), mas perde no
ViSP (1,5–2,0×). Mecanismo: uma máscara de fase pura tem ganho 1 em todas as frequências e
não atenua o ruído longe da linha. No IBIS e no CRISP o V é zero fora da faixa medida; no
ViSP o ruído cobre a janela inteira (±5 Å) e chega ao detector. Um diafragma de campo
(fenda no plano de entrada) limita a janela espectral opticamente.

Variantes (POF do template v2 com o mesmo diafragma, fase treinada, janela 3×5):
  ±200 pm, ±120 pm, ±80 pm   (CRISP mede ±90 pm na grade provisória; IBIS −120 a +160 pm)
Referência: o 62 (sem diafragma), carregado do checkpoint.

Avaliação no teste grande v2 (acurácia) e limite de detecção a 5% de falso alarme em k = 1
(mesmas sementes e conjuntos do 54: sem campo 1500, varredura Quieta+Rede 1000/classe),
por instrumento, como razão B·f₅₀(D²NN)/B·f₅₀(filtro |a|).

Critérios (definidos antes):
  C1  ViSP: razão ≤ 1,2 na melhor variante (no 62: 1,51 em k = 1)
  C2  IBIS e CRISP: razão da melhor variante ≤ a do 62 + 0,05
  Escolha: menor razão média nos três instrumentos, com acurácia ≥ a do 62 − 1 pp.
  A escolhida é salva também como CheckPoints/modelo_d2nn_optico_v2.pt (--modelo 67 no 54).

Uso:
    python Scripts/67_d2nn_optica_diafragma.py
"""

import sys
import shutil
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from Src.config import DEVICE, DTYPE
from Src.model import B3
from Src.dataset import generate_master_dataset, generate_degraded_profiles, INST_NOMES, NOISE_STD, SEED_BASE
from Src.train_utils import X_SCALE_TREINO, MODEL_INST_V2_PATH
from Src.evaluation import (carregar_modelo, MontadorEntrada, prever, prever_logits, softmax,
                            resumo_classificacao, templates_filtro_casado, amplitude_filtro_casado,
                            ajustar_limiares, classificar_filtro_casado, amostrar_varredura_bf, limiar_bf50)
from Src.treino_optico import dados_v2, fase_pof, treinar, INSTR
from Src.report_utils import save_report

DIAFRAGMAS_PM = (200, 120, 80)
JANELA = (3, 5)
MAX_EPOCAS = 80
FAP = 0.05
N_TESTE_GRANDE_POR_CLASSE = 1000
SEED_TESTE_GRANDE = SEED_BASE + 1000
SEED_NULO_DEG, SEED_VAR, SEED_VAR_DEG = SEED_BASE + 4003, SEED_BASE + 4002, SEED_BASE + 4004   # como no 54
NOMES = B3['class_names']
CKPT = MODEL_INST_V2_PATH.parent
REF_62 = CKPT / 'modelo_d2nn_optico_pof_treinado.pt'
MELHOR = CKPT / 'modelo_d2nn_optico_v2.pt'


def loader(V):
    return DataLoader(TensorDataset(torch.tensor(np.asarray(V), dtype=DTYPE, device=DEVICE)),
                      batch_size=B3['batch_size'])


def main():
    t0 = time.time()
    montar = MontadorEntrada(X_SCALE_TREINO, DEVICE)
    dados = dados_v2()
    rng = np.random.default_rng(0)

    # Teste grande v2 e conjuntos de detecção (k = 1)
    (_, y_g, B_g, F_g, *_) = generate_master_dataset(NOISE_STD, SEED_TESTE_GRANDE, 'static',
                                                     n_samples=N_TESTE_GRANDE_POR_CLASSE)
    _, V_g, inst_g = generate_degraded_profiles(B_g, F_g, seed=SEED_TESTE_GRANDE, instrumental=INSTR)
    B0 = np.zeros(1500)
    _, V0, i0 = generate_degraded_profiles(B0, B0, seed=SEED_NULO_DEG, instrumental=INSTR)
    Bv, Fv, _ = amostrar_varredura_bf(1000, SEED_VAR, classes=(0, 1))
    _, Vv, iv = generate_degraded_profiles(Bv, Fv, seed=SEED_VAR_DEG, instrumental=INSTR)
    bf = Bv * Fv

    def b50(esc0, escv):
        out = {}
        for k, nm in enumerate(INST_NOMES):
            thr = np.quantile(esc0[i0 == k], 1 - FAP)
            det = (escv[iv == k] > thr).astype(float)
            out[nm] = limiar_bf50(bf[iv == k], det, rng, n_boot=0, piso=FAP)[0]
        return out

    templ = templates_filtro_casado(INST_NOMES, instrumental=INSTR)
    lim = ajustar_limiares(amplitude_filtro_casado(dados['V_fixo'][dados['idx_train']],
                                                   dados['inst_fixo'][dados['idx_train']], templ, modulo=True),
                           dados['y'][dados['idx_train']])
    acc_fc = resumo_classificacao(y_g, classificar_filtro_casado(
        amplitude_filtro_casado(V_g, inst_g, templ, modulo=True), lim), inst_g, NOMES, INST_NOMES, '', verbose=False)
    b50_fc = b50(amplitude_filtro_casado(V0, i0, templ, modulo=True), amplitude_filtro_casado(Vv, iv, templ, modulo=True))

    def avaliar_modelo(modelo):
        r = resumo_classificacao(y_g, prever(loader(V_g), modelo, montar), inst_g, NOMES, INST_NOMES, '',
                                 verbose=False)
        e0 = 1 - softmax(prever_logits(loader(V0), modelo, montar))[:, 0]
        ev = 1 - softmax(prever_logits(loader(Vv), modelo, montar))[:, 0]
        b = b50(e0, ev)
        return r, b, {nm: b[nm] / b50_fc[nm] for nm in INST_NOMES}

    res = {}
    if REF_62.exists():
        m62, _ = carregar_modelo(REF_62, DEVICE)
        res['62 (sem diafragma)'] = dict(zip(('resumo', 'bf50', 'razao'), avaliar_modelo(m62)))
        res['62 (sem diafragma)'].update({'diafragma_pm': None, 'caminho': str(REF_62)})
        del m62
    else:
        print(f'  ⚠ {REF_62.name} não encontrado — rode o 62 antes; seguindo sem a referência')

    for d in DIAFRAGMAS_PM:
        nome = f'±{d} pm'
        modelo, info = treinar(nome, fase_pof(montar, d), True, JANELA, montar, dados, MAX_EPOCAS, diafragma_pm=d)
        r, b, raz = avaliar_modelo(modelo)
        caminho = CKPT / f'modelo_d2nn_optico_diafragma_{d}pm.pt'
        torch.save({'model_state_dict': modelo.state_dict(), 'arquitetura': 'D2NNOptica', 'janela': list(JANELA),
                    'diafragma_pm': d, 'variante': f'pof_treinado_diafragma_{d}pm', 'X_scale': X_SCALE_TREINO,
                    'modelo_instrumental': 'v2', 'info': {k: v for k, v in info.items() if k != 'historico'}},
                   caminho)
        res[nome] = {'resumo': r, 'bf50': b, 'razao': raz, 'diafragma_pm': d, 'caminho': str(caminho),
                     'treino': info}
        del modelo
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    print(f'\n{"="*104}\nTESTE GRANDE v2 E LIMITE DE DETECÇÃO (5% FA, k = 1) — razão B·f₅₀ D²NN / filtro |a|\n{"="*104}')
    print(f'  {"Variante":22s} {"Geral":>7s} {"Rede":>6s} {"Mancha":>7s}' + ''.join(f'{n:>14s}' for n in INST_NOMES)
          + f'{"média":>8s}')
    print(f'  {"filtro |a|":22s} {acc_fc["acuracia"]:6.2f}% {acc_fc["por_classe"][1]:5.1f}% {acc_fc["por_classe"][2]:6.1f}%'
          + ''.join(f'{b50_fc[n]:>9.1f} G    ' for n in INST_NOMES))
    for nome, e in res.items():
        r, raz = e['resumo'], e['razao']
        e['razao_media'] = float(np.nanmean(list(raz.values())))
        print(f'  {nome:22s} {r["acuracia"]:6.2f}% {r["por_classe"][1]:5.1f}% {r["por_classe"][2]:6.1f}%'
              + ''.join(f'{e["bf50"][n]:>7.1f} ({raz[n]:.2f})' for n in INST_NOMES) + f'{e["razao_media"]:>8.2f}')

    ref = res.get('62 (sem diafragma)')
    cand = {k: v for k, v in res.items() if v['diafragma_pm'] is not None and
            (ref is None or v['resumo']['acuracia'] >= ref['resumo']['acuracia'] - 1.0)}
    if not cand:
        print('\n  Nenhuma variante com diafragma manteve a acurácia (≥ 62 − 1 pp).')
        escolha = None
    else:
        escolha = min(cand, key=lambda k: cand[k]['razao_media'])
        e = res[escolha]
        c1 = e['razao']['ViSP/DKIST'] <= 1.2
        c2 = ref is None or all(e['razao'][n] <= ref['razao'][n] + 0.05 for n in ('IBIS/DST', 'CRISP/SST'))
        print(f'\n  Escolhida: {escolha} (razão média {e["razao_media"]:.2f})')
        print(f'  C1 ViSP ≤ 1,2: {e["razao"]["ViSP/DKIST"]:.2f} → {"ATENDIDO" if c1 else "NÃO ATENDIDO"}')
        print(f'  C2 IBIS e CRISP sem piora (≤ 62 + 0,05): '
              + ', '.join(f'{n} {e["razao"][n]:.2f}' + (f' (62: {ref["razao"][n]:.2f})' if ref else '')
                          for n in ('IBIS/DST', 'CRISP/SST')) + f' → {"ATENDIDO" if c2 else "NÃO ATENDIDO"}')
        shutil.copy2(e['caminho'], MELHOR)
        print(f'  Copiada para {MELHOR.name} (use --modelo 67 no 54)')
        res[escolha]['criterios'] = {'C1': bool(c1), 'C2': bool(c2)}
    print(f'\n  Tempo total: {(time.time() - t0) / 60:.1f} min')
    save_report('67_d2nn_optica_diafragma', {
        'diafragmas_pm': list(DIAFRAGMAS_PM), 'janela': list(JANELA), 'fap': FAP,
        'filtro_modulo': {'resumo': acc_fc, 'bf50_k1': b50_fc}, 'variantes': res, 'escolhida': escolha,
        'tempo_min': (time.time() - t0) / 60})


if __name__ == '__main__':
    main()
