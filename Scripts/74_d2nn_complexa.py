"""
74_d2nn_complexa.py — D²NN óptica com máscara de AMPLITUDE E FASE no plano de Fourier.

Motivação (script 73): no ViSP a perda da D²NN óptica vem da máscara só de fase (H2). O POF
(H = S*/|S|) tem ganho 1 em todas as frequências; mesmo calculado para o ViSP e com diafragma,
não passa de 1,56× o filtro casado. A máscara casada complexa (H = S*) reproduz o filtro
(0,99–1,00×). Aqui: uma D²NN com máscara complexa treinável, UMA máscara para os três
instrumentos (protocolo do 62/67: mesmo dataset v2, ruído renovado por época, janela 3 × 5,
dois limiares ordinais).

Máscara: t = a·e^{iφ}, 0 < a < 1 (passiva: SLM de fase + máscara de absorção), iniciada em
H = S*/max|S| do template médio v2 (o mesmo template do POF do 62/67).

Variantes:
  complexa fixa              : máscara casada do template médio, só os limiares treinados
  complexa treinada          : amplitude e fase treinadas, sem diafragma
  complexa treinada ±200 pm  : idem, com o diafragma do 67
Referências: 67 (checkpoint) e filtro |a|.

Avaliação (como no 67): teste grande v2 (acurácia) e B·f₅₀ a 5% de falso alarme em k = 1 por
instrumento (sem campo 1500, varredura Quieta + Rede 1000/classe; sementes do 54), agora com o
estimador físico (β = 1, script 70); razão B·f₅₀(D²NN)/B·f₅₀(filtro |a|).
Transmissão da máscara: fração da potência que passa, ⟨a²⟩ na pupila (custo de luz da amplitude).

Critérios (definidos antes), na variante escolhida:
  C1  ViSP: razão ≤ 1,10 (67: 1,20 em k = 1 no 54; 1,32 no 73)
  C2  IBIS ≤ 1,10 e CRISP ≤ a do 67 + 0,05
  Escolha: menor razão média nos três instrumentos, com acurácia ≥ a do 67 − 1 pp. Copiada para
  CheckPoints/modelo_d2nn_complexo_v2.pt (--modelo 74 no 54).

Uso:
    python Scripts/74_d2nn_complexa.py
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
from Src.model import B3, PUPIL_T
from Src.dataset import generate_master_dataset, generate_degraded_profiles, INST_NOMES, NOISE_STD, SEED_BASE
from Src.train_utils import X_SCALE_TREINO, MODEL_INST_V2_PATH
from Src.evaluation import (carregar_modelo, MontadorEntrada, prever, prever_logits, softmax,
                            resumo_classificacao, templates_filtro_casado, amplitude_filtro_casado,
                            ajustar_limiares, classificar_filtro_casado, amostrar_varredura_bf,
                            limiar_bf50_fisico)
from Src.treino_optico import dados_v2, mascara_casada, treinar, INSTR
from Src.report_utils import save_report

JANELA = (3, 5)
MAX_EPOCAS = 80
FAP = 0.05
N_TESTE_GRANDE_POR_CLASSE = 1000
SEED_TESTE_GRANDE = SEED_BASE + 1000
SEED_NULO_DEG, SEED_VAR, SEED_VAR_DEG = SEED_BASE + 4003, SEED_BASE + 4002, SEED_BASE + 4004   # como no 54
NOMES = B3['class_names']
CKPT = MODEL_INST_V2_PATH.parent
REF_67 = CKPT / 'modelo_d2nn_optico_v2.pt'
MELHOR = CKPT / 'modelo_d2nn_complexo_v2.pt'
VARIANTES = [('complexa fixa', False, None), ('complexa treinada', True, None),
             ('complexa treinada ±200 pm', True, 200)]


def loader(V):
    return DataLoader(TensorDataset(torch.tensor(np.asarray(V), dtype=DTYPE, device=DEVICE)),
                      batch_size=B3['batch_size'])


def transmissao(modelo):
    pup = PUPIL_T.abs() > 0
    a = modelo.mascara.amplitude().detach()
    return float((a[pup] ** 2).mean())


def main():
    t0 = time.time()
    montar = MontadorEntrada(X_SCALE_TREINO, DEVICE)
    dados = dados_v2()

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
            out[nm] = limiar_bf50_fisico(bf[iv == k], det, None, n_boot=0, fap=FAP)[0]
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
    if REF_67.exists():
        m67, _ = carregar_modelo(REF_67, DEVICE)
        res['67 (só fase, ±200 pm)'] = dict(zip(('resumo', 'bf50', 'razao'), avaliar_modelo(m67)))
        res['67 (só fase, ±200 pm)'].update({'diafragma_pm': 200, 'caminho': str(REF_67), 'transmissao': 1.0})
        del m67
    else:
        print(f'  ⚠ {REF_67.name} não encontrado — seguindo sem a referência')

    for nome, treinar_mascara, d in VARIANTES:
        fase, amp = mascara_casada(montar, d)
        modelo, info = treinar(nome, fase, treinar_mascara, JANELA, montar, dados, MAX_EPOCAS,
                               diafragma_pm=d, amp_init=amp)
        r, b, raz = avaliar_modelo(modelo)
        tag = nome.replace(' ', '_').replace('±', '').replace('_pm', 'pm')
        caminho = CKPT / f'modelo_d2nn_{tag}.pt'
        torch.save({'model_state_dict': modelo.state_dict(), 'arquitetura': 'D2NNOpticaComplexa',
                    'janela': list(JANELA), 'diafragma_pm': d, 'variante': tag, 'X_scale': X_SCALE_TREINO,
                    'modelo_instrumental': 'v2', 'info': {k: v for k, v in info.items() if k != 'historico'}},
                   caminho)
        res[nome] = {'resumo': r, 'bf50': b, 'razao': raz, 'diafragma_pm': d, 'caminho': str(caminho),
                     'treino': info, 'transmissao': transmissao(modelo), 'treinada': treinar_mascara}
        del modelo
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    print(f'\n{"="*112}\nTESTE GRANDE v2 E LIMITE DE DETECÇÃO (5% FA, k = 1, estimador físico) — razão B·f₅₀ D²NN / filtro |a|\n{"="*112}')
    print(f'  {"Variante":28s} {"Geral":>7s} {"Rede":>6s} {"Mancha":>7s}' + ''.join(f'{n:>14s}' for n in INST_NOMES)
          + f'{"média":>8s} {"transm.":>8s}')
    print(f'  {"filtro |a|":28s} {acc_fc["acuracia"]:6.2f}% {acc_fc["por_classe"][1]:5.1f}% {acc_fc["por_classe"][2]:6.1f}%'
          + ''.join(f'{b50_fc[n]:>9.1f} G    ' for n in INST_NOMES))
    for nome, e in res.items():
        r, raz = e['resumo'], e['razao']
        e['razao_media'] = float(np.nanmean(list(raz.values())))
        print(f'  {nome:28s} {r["acuracia"]:6.2f}% {r["por_classe"][1]:5.1f}% {r["por_classe"][2]:6.1f}%'
              + ''.join(f'{e["bf50"][n]:>7.1f} ({raz[n]:.2f})' for n in INST_NOMES)
              + f'{e["razao_media"]:>8.2f} {100 * e["transmissao"]:7.1f}%')

    ref = res.get('67 (só fase, ±200 pm)')
    cand = {k: v for k, v in res.items() if v.get('treinada') and
            (ref is None or v['resumo']['acuracia'] >= ref['resumo']['acuracia'] - 1.0)}
    escolha = None
    if not cand:
        print('\n  Nenhuma variante treinada manteve a acurácia (≥ 67 − 1 pp).')
    else:
        escolha = min(cand, key=lambda k: cand[k]['razao_media'])
        e = res[escolha]
        c1 = e['razao']['ViSP/DKIST'] <= 1.10
        c2 = e['razao']['IBIS/DST'] <= 1.10 and (ref is None or e['razao']['CRISP/SST'] <= ref['razao']['CRISP/SST'] + 0.05)
        print(f'\n  Escolhida: {escolha} (razão média {e["razao_media"]:.2f}; transmissão {100 * e["transmissao"]:.1f}%)')
        print(f'  C1 ViSP ≤ 1,10: {e["razao"]["ViSP/DKIST"]:.2f} → {"ATENDIDO" if c1 else "NÃO ATENDIDO"}')
        print(f'  C2 IBIS ≤ 1,10 e CRISP ≤ 67 + 0,05: IBIS {e["razao"]["IBIS/DST"]:.2f}, CRISP {e["razao"]["CRISP/SST"]:.2f}'
              + (f' (67: {ref["razao"]["CRISP/SST"]:.2f})' if ref else '') + f' → {"ATENDIDO" if c2 else "NÃO ATENDIDO"}')
        shutil.copy2(e['caminho'], MELHOR)
        print(f'  Copiada para {MELHOR.name} (use --modelo 74 no 54)')
        e['criterios'] = {'C1': bool(c1), 'C2': bool(c2)}
    print(f'\n  Tempo total: {(time.time() - t0) / 60:.1f} min')
    save_report('74_d2nn_complexa', {
        'janela': list(JANELA), 'fap': FAP, 'estimador': 'limiar_bf50_fisico (beta = 1)',
        'filtro_modulo': {'resumo': acc_fc, 'bf50_k1': b50_fc}, 'variantes': res, 'escolhida': escolha,
        'tempo_min': (time.time() - t0) / 60})


if __name__ == '__main__':
    main()
