"""
61_ablacao_optica.py — Quanto da classificação vem da parte ÓPTICA da D²NN?

Migra e substitui as Células 36, 36B, 36_FIX e 37 do Colab.

Motivação: no modelo do Colab, as três máscaras convergiram idênticas e a
correção de fase aprendida tinha std de 0,0016 rad sobre o vórtice m = 1
(Célula 36_FIX): "filtro de vórtice estático + MLP espectral". Como as três
CamadaFase equivalem a uma máscara só (achado 5), o que importa é a fase TOTAL
φ = Σ (fase_base + Δφ). Se o mesmo valer para os modelos atuais, o artigo B
precisa dizer quanto do resultado é óptica e quanto é a cabeça eletrônica.

Parte 1 — fases aprendidas (sem dados): por modelo e por camada, std e máx |Δφ|
dentro da pupila, diferença entre camadas (identidade) e Δφ total.

Parte 2 — ablação com a cabeça eletrônica FIXA (Conv1d + BN + MLP do checkpoint):
  V0  treinado            : como salvo
  V1  vórtice estático    : Δφ = 0 nas três camadas (só o vórtice m = 1 de partida)
  V2  só pupila           : fase total = 0 (sistema 4f sem modulação de fase)
  V3  fase aleatória      : camada 1 com fase uniforme em [−π, π) (semente fixa), Δφ = 0
  V4  Δφ × 10             : correção aprendida amplificada 10× (sensibilidade)

Leitura (critério definido antes):
  - V1 ≈ V0 (|Δ| < 1 pp)  → a fase APRENDIDA não contribui; a óptica efetiva é
    o vórtice de inicialização.
  - V2 e V3 ≈ V0         → a cabeça classifica com qualquer óptica; o papel da
    óptica é só levar o espectro ao detector.
  - V2 << V0             → o vórtice importa. (V3 << V0 sozinho não mede o vórtice: uma
    fase aleatória é um difusor que embaralha o espectrograma.)
  Ressalva: trocar a máscara muda a distribuição na entrada da cabeça, que foi
  treinada com a máscara original (BatchNorm incluído). Uma queda em V2/V3 pode ser
  só descasamento de distribuição; ausência de queda é o resultado forte.

Conjuntos: 57 e 59 no teste grande v2 (o mesmo do 57/59); 47 no teste grande
legado (o mesmo do 47–52). Sem treino; não toca em checkpoints.

Resultado (2026-10-22): nos três modelos V1 ≈ V2 ≈ V0 (|Δ| ≤ 0,43 pp, p ≥ 0,17):
nem a fase aprendida nem o vórtice contribuem; o sistema efetivo é imageamento 4f
fixo + cabeça eletrônica. As três camadas recebem o mesmo gradiente (a perda só
depende da fase total) e permanecem idênticas. Motiva o script 62.

Uso:
    python Scripts/61_ablacao_optica.py
"""

import sys
import copy
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from Src.config import DEVICE, DTYPE, FIGURES_DIR
from Src.model import B3
from Src.optics import FATOR_RAMPA
from Src.physics import INSTRUMENTAL_V2
from Src.dataset import generate_master_dataset, generate_degraded_profiles, NOISE_STD, SEED_BASE
from Src.train_utils import X_SCALE_TREINO, MODEL_DEG_AUG_PATH, MODEL_INST_V2_PATH
from Src.evaluation import (carregar_modelo, MontadorEntrada, prever, resumo_classificacao,
                            mcnemar)
from Src.report_utils import save_report

N_TESTE_GRANDE_POR_CLASSE = 1000
SEED_TESTE_GRANDE         = SEED_BASE + 1000
SEED_FASE_ALEATORIA       = 12345
NOMES = B3['class_names']
MODELOS = (
    ('47', MODEL_DEG_AUG_PATH, 'legado', 'D²NN 47 (intensidade, legado)'),
    ('57', MODEL_INST_V2_PATH, 'v2', 'D²NN 57 (intensidade, v2)'),
    ('59', MODEL_INST_V2_PATH.with_name('modelo_d2nn_homodino_v2.pt'), 'v2',
     'D²NN 59 (homodina, v2, polaridade +)'),
)
VARIANTES = ('V0 treinado', 'V1 vórtice estático', 'V2 só pupila', 'V3 fase aleatória', 'V4 Δφ × 10')
CAMADAS = ('camada1', 'camada2', 'camada3')


def teste_grande(instrumental):
    (_, y, B, F, *_) = generate_master_dataset(NOISE_STD, SEED_TESTE_GRANDE, 'static',
                                               n_samples=N_TESTE_GRANDE_POR_CLASSE)
    _, V, _ = generate_degraded_profiles(B, F, seed=SEED_TESTE_GRANDE,
                                         instrumental=INSTRUMENTAL_V2 if instrumental == 'v2' else None)
    loader = DataLoader(TensorDataset(torch.tensor(V, dtype=DTYPE, device=DEVICE)),
                        batch_size=B3['batch_size'])
    return y, loader


def estatisticas_fase(modelo):
    """Δφ por camada e total, dentro da pupila (rad)."""
    pup = (modelo.camada1.pupila.detach().abs() > 0).cpu().numpy()
    dphi = {c: getattr(modelo, c).delta_phi.detach().cpu().numpy() for c in CAMADAS}
    base = {c: getattr(modelo, c).fase_base.detach().cpu().numpy() for c in CAMADAS}
    out = {'pixels_pupila': int(pup.sum())}
    for c in CAMADAS:
        d = dphi[c][pup]
        out[c] = {'std_dphi': float(d.std()), 'max_abs_dphi': float(np.abs(d).max()),
                  'std_fase_base': float(base[c][pup].std())}
    out['max_dif_entre_camadas'] = float(max(np.abs(dphi[a][pup] - dphi[b][pup]).max()
                                             for a, b in (('camada1', 'camada2'),
                                                          ('camada1', 'camada3'),
                                                          ('camada2', 'camada3'))))
    total = sum(dphi.values())
    out['dphi_total'] = {'std': float(total[pup].std()), 'max_abs': float(np.abs(total[pup]).max())}
    return out, total, pup


def variante(modelo, nome):
    m = copy.deepcopy(modelo)
    with torch.no_grad():
        if nome.startswith('V1'):
            for c in CAMADAS:
                getattr(m, c).delta_phi.zero_()
        elif nome.startswith('V2'):
            for c in CAMADAS:
                getattr(m, c).delta_phi.zero_()
                getattr(m, c).fase_base.zero_()
        elif nome.startswith('V3'):
            g = torch.Generator(device='cpu').manual_seed(SEED_FASE_ALEATORIA)
            fb = m.camada1.fase_base
            rnd = (torch.rand(fb.shape, generator=g) * 2 - 1) * np.pi
            fb.copy_(rnd.to(fb.device, fb.dtype))
            for c in CAMADAS:
                getattr(m, c).delta_phi.zero_()
        elif nome.startswith('V4'):
            for c in CAMADAS:
                getattr(m, c).delta_phi.mul_(10.0)
    m.eval()
    return m


def main():
    conjuntos = {}
    rel = {'variantes': list(VARIANTES), 'modelos': {}}
    totais = {}
    for chave, caminho, instr, titulo in MODELOS:
        if not caminho.exists():
            print(f'\n{titulo}: checkpoint não encontrado ({caminho.name}) — pulando')
            continue
        modelo, meta = carregar_modelo(caminho, DEVICE)
        montar = MontadorEntrada(float(meta.get('X_scale', X_SCALE_TREINO)), DEVICE,
                                 fator_rampa=tuple(meta.get('fator_rampa', FATOR_RAMPA)))
        if instr not in conjuntos:
            conjuntos[instr] = teste_grande(instr)
        y, loader = conjuntos[instr]

        est, total, pup = estatisticas_fase(modelo)
        totais[chave] = (total, pup)
        print(f'\n{"="*84}\n{titulo}  |  {meta.get("arquitetura")}  |  teste grande {instr} '
              f'({len(y)} perfis)\n{"="*84}')
        print(f'  Fases (rad, dentro da pupila, {est["pixels_pupila"]} pixels):')
        for c in CAMADAS:
            e = est[c]
            print(f'    {c}: std(Δφ) = {e["std_dphi"]:.2e}  máx|Δφ| = {e["max_abs_dphi"]:.2e}  '
                  f'std(fase_base) = {e["std_fase_base"]:.3f}')
        print(f'    máx diferença de Δφ entre camadas: {est["max_dif_entre_camadas"]:.2e}')
        print(f'    Δφ total (Σ camadas): std = {est["dphi_total"]["std"]:.2e}  '
              f'máx = {est["dphi_total"]["max_abs"]:.2e}')
        if 'beta_ref' in meta:
            print(f'    β (campo de referência) = {float(meta["beta_ref"]):+.4f}')

        print(f'\n  {"Variante":24s} {"Geral":>8s}' + ''.join(f'{n[:8]:>10s}' for n in NOMES)
              + f'{"Δ vs V0":>10s}{"p":>10s}')
        res, pred0 = {}, None
        for nome in VARIANTES:
            m = variante(modelo, nome)
            pred = prever(loader, m, montar)
            r = resumo_classificacao(y, pred, None, NOMES, [], '', verbose=False)
            if pred0 is None:
                pred0 = pred
                mc = None
            else:
                mc = mcnemar(y, pred, pred0)
            res[nome] = {'acuracia': r['acuracia'], 'por_classe': r['por_classe'],
                         'matriz': r['matriz'], 'mcnemar_vs_V0': mc}
            print(f'  {nome:24s} {r["acuracia"]:>7.2f}%' + ''.join(f'{v:>9.1f}%' for v in r['por_classe'])
                  + (f'{mc["diferenca_pp"]:>+9.2f}{mc["p_valor"]:>10.2g}' if mc else f'{"—":>10s}{"":>10s}'))
            del m
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

        d1 = res['V1 vórtice estático']['acuracia'] - res['V0 treinado']['acuracia']
        d2 = res['V2 só pupila']['acuracia'] - res['V0 treinado']['acuracia']
        d3 = res['V3 fase aleatória']['acuracia'] - res['V0 treinado']['acuracia']
        print(f'\n  Leitura: fase aprendida {"NÃO contribui" if abs(d1) < 1.0 else "contribui"} '
              f'(V1 − V0 = {d1:+.2f} pp); vórtice '
              f'{"irrelevante (só pupila ≈ treinado)" if abs(d2) < 1.0 else "importa"} '
              f'(V2 − V0 = {d2:+.2f} pp). Fase aleatória: V3 − V0 = {d3:+.2f} pp '
              f'(difusor que embaralha o espectrograma + descasamento com a cabeça; não mede o vórtice)')
        rel['modelos'][chave] = {'titulo': titulo, 'instrumental': instr,
                                 'arquitetura': meta.get('arquitetura'),
                                 'beta_ref': meta.get('beta_ref'), 'fases': est, 'ablacao': res}
        del modelo

    # --- Figura: Δφ total de cada modelo ---
    if totais:
        fig, axs = plt.subplots(1, len(totais), figsize=(6 * len(totais), 4.2),
                                constrained_layout=True, squeeze=False)
        fig.suptitle('Correção de fase aprendida (Δφ total = Σ camadas) — dentro da pupila',
                     fontsize=12, fontweight='bold')
        for ax, (chave, (total, pup)) in zip(axs[0], totais.items()):
            t = np.where(pup, total, np.nan)
            lim = np.nanmax(np.abs(t)) or 1.0
            im = ax.imshow(t, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim,
                           interpolation='nearest')
            ax.set_title(f'D²NN {chave}: máx |Δφ| = {lim:.2e} rad', fontsize=10)
            ax.set_xlabel('frequência espectral (pixel)'); ax.set_ylabel('frequência espacial (linha)')
            fig.colorbar(im, ax=ax, label='Δφ [rad]')
        out = FIGURES_DIR / 'cel61_fase_aprendida.png'
        fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
        rel['figura'] = str(out)
        print(f'\n  Figura: {out.name}')

    save_report('61_ablacao_optica', rel)


if __name__ == '__main__':
    main()
