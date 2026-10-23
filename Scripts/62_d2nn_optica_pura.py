"""
62_d2nn_optica_pura.py — D²NN PURAMENTE ÓPTICA: máscara de fase + um detector.

Motivação (script 61): nos modelos 47, 57 e 59 a fase aprendida não contribui
(Δφ ~ 2–10 mrad; zerar a fase ou trocar o vórtice por uma pupila simples muda
< 0,5 pp). O sistema efetivo é imageamento 4f fixo + cabeça eletrônica com ~4,2 M
parâmetros, que compensa qualquer óptica e não deixa gradiente útil para as fases.

Física da leitura: no dataset as três classes têm a MESMA forma de V (achado 18).
Com óptica linear e detecção |E|², o padrão no detector só muda de escala (c²)
entre classes; razões de energia entre regiões (leitura do Lin et al. 2018) não
carregam informação. Toda leitura óptica se reduz a UM escalar, e o ótimo é
concentrar a projeção casada num ponto: |E|² = |a|², equivalente ao filtro |a|.
Uma máscara de fase pura faz isso com o filtro casado de fase (POF):
H = S*/|S| ⇒ saída = IFFT(|S|), um pico na origem; o ruído branco continua branco.

Modelo: X (64 × N_LAM) → uma CamadaFase (Fourier, dentro da pupila) → |E|² →
energia E numa janela pequena centrada no pico → logits_k = w_k·z + b_k,
z = (log E − μ)/s. A "eletrônica" são 6 números (dois limiares ordinais) — um
comparador. Invariante à polaridade por construção (|E|² é par em V).

Variantes:
  pof_fixo     : fase = POF analítico do template médio v2; só w, b treinados
  pof_treinado : parte do POF; fase também treinada

Referências no teste grande v2 (o mesmo do 57/59): filtro casado |a| (limite para
leitura de intensidade, polaridade desconhecida), filtro com sinal e D²NN 57.

Critérios (definidos antes):
  C1  pof_treinado ≥ filtro |a| − 3 pp e Mancha ≥ 95%   (D²NN óptica útil)
  C2  pof_treinado > 57 (McNemar, p < 0,05)              (melhor que o híbrido)
  C3  |acc(+V) − acc(−V)| < 0,5 pp                        (invariância, sanidade)

Salva CheckPoints/modelo_d2nn_optico_<variante>.pt (formato próprio; não é lido
por carregar_modelo). Está no PULAR do 00_run_all.py.

Resultado (2026-10-22, janela 3 × 5): pof_treinado 86,67% contra 86,13% do filtro |a|
(+0,53 pp, p = 0,25) e 76,40% do 57 (+10,27 pp, p = 1,8e-35); pof_fixo 83,10%;
invariância exata à polaridade; std(Δφ) = 1,18 rad. C1, C2 e C3 atendidos.
A classe D2NNOptica está em Src/model.py (carregar_modelo a reconhece).

Uso:
    python Scripts/62_d2nn_optica_pura.py
    python Scripts/62_d2nn_optica_pura.py --janela 3 5 --max-epocas 80
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from Src.config import DEVICE, DTYPE, FIGURES_DIR, SEED
from Src.model import B3, PUPIL_T, D2NNOptica
from Src.physics import INSTRUMENTAL_V2
from Src.dataset import (generate_master_dataset, generate_degraded_profiles,
                         INST_NOMES, INST_PROBS, NOISE_STD, SEED_BASE)
from Src.train_utils import split_indices, X_SCALE_TREINO, MODEL_INST_V2_PATH
from Src.evaluation import (carregar_modelo, MontadorEntrada, prever, resumo_classificacao,
                            templates_filtro_casado, amplitude_filtro_casado,
                            ajustar_limiares, classificar_filtro_casado, mcnemar)
from Src.report_utils import save_report

INSTR                     = INSTRUMENTAL_V2
BOUNDARY_WEIGHT           = 0.5
PESOS_CLASSE              = [1.5, 1.5, 1.0]          # os mesmos do train_model
SEED_DEG_FIXO             = SEED + 2
SEED_DEG_EPOCA            = SEED + 10_000
N_TESTE_GRANDE_POR_CLASSE = 1000
SEED_TESTE_GRANDE         = SEED_BASE + 1000
LR_FASE, LR_LEITURA       = 1e-2, 1e-2
PACIENCIA                 = 15
NOMES = B3['class_names']
CKPT_DIR = MODEL_INST_V2_PATH.parent


def fase_pof(montar):
    """φ = −arg F{X_template} (template médio v2 ponderado por INST_PROBS), centrada."""
    templ = templates_filtro_casado(INST_NOMES, instrumental=INSTR)
    p = np.asarray([INST_PROBS[n] for n in INST_NOMES] if isinstance(INST_PROBS, dict)
                   else INST_PROBS, dtype=float)
    t = sum(p[k] * templ[k] for k in range(len(INST_NOMES))) / p.sum()
    Xt = montar(torch.tensor(t[None, :], dtype=DTYPE, device=DEVICE))[0]
    S = torch.fft.fft2(Xt.to(torch.complex64), norm='ortho')
    return torch.fft.fftshift(-torch.angle(S), dim=(-2, -1)).to(DTYPE)


def avaliar(modelo, loader, montar):
    modelo.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            out.append(modelo(montar(batch[0])).argmax(1).cpu().numpy())
    return np.concatenate(out)


def perda_lote(modelo, montar, batch, pesos):
    V, y, _, _, w = batch
    logits = modelo(montar(V))
    return (Fnn.cross_entropy(logits, y, weight=pesos, reduction='none') * w).mean(), logits, y


def treinar(nome, fase_init, treinar_fase, janela, montar, loaders, renovar, max_epocas):
    train_loader, val_loader = loaders
    modelo = D2NNOptica(fase_init, janela, treinar_fase).to(DEVICE)
    # normalização de z com o estado inicial (energias de treino)
    with torch.no_grad():
        z = torch.cat([torch.log(modelo.energia(montar(b[0])) + 1e-12) for b in train_loader])
        modelo.mu.fill_(float(z.mean())); modelo.s.fill_(float(z.std()) or 1.0)
    grupos = [{'params': [modelo.w, modelo.b], 'lr': LR_LEITURA}]
    if treinar_fase:
        grupos.append({'params': [modelo.mascara.delta_phi], 'lr': LR_FASE})
    opt = torch.optim.Adam(grupos)
    pesos = torch.tensor(PESOS_CLASSE, dtype=DTYPE, device=DEVICE)
    melhor, estado, epoca_melhor, sem_melhora = float('inf'), None, 0, 0
    hist = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    print(f'\n  Treino — {nome}  (fase {"treinável" if treinar_fase else "fixa"}, janela {janela})')
    for ep in range(1, max_epocas + 1):
        renovar(ep)
        modelo.train()
        tl = []
        for batch in train_loader:
            opt.zero_grad()
            loss, _, _ = perda_lote(modelo, montar, batch, pesos)
            loss.backward(); opt.step(); tl.append(loss.item())
        modelo.eval()
        vl, acertos, n = [], 0, 0
        with torch.no_grad():
            for batch in val_loader:
                loss, logits, y = perda_lote(modelo, montar, batch, pesos)
                vl.append(loss.item() * len(y)); acertos += int((logits.argmax(1) == y).sum()); n += len(y)
        vloss, vacc = sum(vl) / n, 100 * acertos / n
        for k, v in (('train_loss', np.mean(tl)), ('val_loss', vloss), ('val_acc', vacc)):
            hist[k].append(float(v))
        marca = ''
        if vloss < melhor:
            melhor, epoca_melhor, sem_melhora, marca = vloss, ep, 0, ' ★'
            estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sem_melhora += 1
        if ep == 1 or ep % 5 == 0 or marca:
            print(f'    época {ep:3d}  loss T {np.mean(tl):.4f}  loss V {vloss:.4f}  acc V {vacc:5.1f}%{marca}')
        if sem_melhora >= PACIENCIA:
            print(f'    parada antecipada na época {ep}')
            break
    modelo.load_state_dict(estado)
    dphi = modelo.mascara.delta_phi.detach()
    pup = PUPIL_T.abs() > 0
    print(f'    melhor época {epoca_melhor} (val loss {melhor:.4f}); std(Δφ) na pupila = '
          f'{float(dphi[pup].std()):.3e} rad')
    return modelo, {'epoca_melhor': epoca_melhor, 'melhor_val_loss': melhor, 'historico': hist,
                    'std_dphi': float(dphi[pup].std()), 'max_abs_dphi': float(dphi[pup].abs().max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--janela', type=int, nargs=2, default=[3, 5], help='linhas colunas (ímpares)')
    ap.add_argument('--max-epocas', type=int, default=80)
    args = ap.parse_args()
    janela = tuple(args.janela)
    t0 = time.time()
    montar = MontadorEntrada(X_SCALE_TREINO, DEVICE)

    # --- Dados (protocolo do 57) ---
    (_, y, B_all, F_all, B_norm, F_norm, boundary) = generate_master_dataset(NOISE_STD, SEED_BASE, 'static')
    _, V_fixo, inst_fixo = generate_degraded_profiles(B_all, F_all, seed=SEED_DEG_FIXO, instrumental=INSTR)
    idx_train, idx_val, _ = split_indices(y, B3, SEED)
    w = np.where(boundary, BOUNDARY_WEIGHT, 1.0).astype(np.float32)
    V_t = torch.tensor(V_fixo, dtype=DTYPE, device=DEVICE)
    ds = TensorDataset(V_t, torch.tensor(y, dtype=torch.long, device=DEVICE),
                       torch.tensor(B_norm, dtype=DTYPE, device=DEVICE),
                       torch.tensor(F_norm, dtype=DTYPE, device=DEVICE),
                       torch.tensor(w, dtype=DTYPE, device=DEVICE))
    train_loader = DataLoader(torch.utils.data.Subset(ds, idx_train), batch_size=B3['batch_size'],
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(torch.utils.data.Subset(ds, idx_val), batch_size=B3['batch_size'])
    idx_train_t = torch.tensor(idx_train, device=DEVICE)

    def renovar(ep):
        _, V_novo, _ = generate_degraded_profiles(B_all[idx_train], F_all[idx_train],
                                                  seed=SEED_DEG_EPOCA + ep, instrumental=INSTR)
        V_t[idx_train_t] = torch.tensor(V_novo, dtype=DTYPE, device=DEVICE)

    (_, y_g, B_g, F_g, *_) = generate_master_dataset(NOISE_STD, SEED_TESTE_GRANDE, 'static',
                                                     n_samples=N_TESTE_GRANDE_POR_CLASSE)
    _, V_g, inst_g = generate_degraded_profiles(B_g, F_g, seed=SEED_TESTE_GRANDE, instrumental=INSTR)

    def loader(V):
        return DataLoader(TensorDataset(torch.tensor(np.asarray(V), dtype=DTYPE, device=DEVICE)),
                          batch_size=B3['batch_size'])

    # --- Referências ---
    templ = templates_filtro_casado(INST_NOMES, instrumental=INSTR)
    pred, res = {}, {}
    for nome, mod in (('filtro |a|', True), ('filtro com sinal', False)):
        lim = ajustar_limiares(amplitude_filtro_casado(V_fixo[idx_train], inst_fixo[idx_train], templ, modulo=mod),
                               y[idx_train])
        pred[nome] = classificar_filtro_casado(amplitude_filtro_casado(V_g, inst_g, templ, modulo=mod), lim)
    if MODEL_INST_V2_PATH.exists():
        m57, meta = carregar_modelo(MODEL_INST_V2_PATH, DEVICE)
        pred['D²NN 57 (híbrida)'] = prever(loader(V_g), m57, MontadorEntrada(float(meta.get('X_scale', X_SCALE_TREINO)), DEVICE))
        del m57

    # --- Treino das variantes ---
    phi_pof = fase_pof(montar)
    treinos = {}
    for nome, treinar_fase in (('pof_fixo', False), ('pof_treinado', True)):
        modelo, info = treinar(nome, phi_pof, treinar_fase, janela, montar,
                               (train_loader, val_loader), renovar, args.max_epocas)
        chave = f'D²NN óptica ({nome})'
        pred[chave] = avaliar(modelo, loader(V_g), montar)
        pred_neg = avaliar(modelo, loader(-V_g), montar)
        info['acc_menosV'] = float(100 * (pred_neg == y_g).mean())
        info['inalteradas_pct'] = float(100 * (pred_neg == pred[chave]).mean())
        treinos[nome] = info
        torch.save({'model_state_dict': modelo.state_dict(), 'arquitetura': 'D2NNOptica',
                    'janela': list(janela), 'variante': nome, 'X_scale': X_SCALE_TREINO,
                    'modelo_instrumental': 'v2', 'info': {k: v for k, v in info.items() if k != 'historico'}},
                   CKPT_DIR / f'modelo_d2nn_optico_{nome}.pt')
        del modelo
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    # --- Tabela ---
    print(f'\n{"="*90}\nTESTE GRANDE v2 ({len(y_g)} perfis)\n{"="*90}')
    print(f'  {"Método":34s} {"Geral":>8s}' + ''.join(f'{n[:8]:>10s}' for n in NOMES)
          + ''.join(f'{n[:10]:>12s}' for n in INST_NOMES))
    for nome, p in pred.items():
        r = resumo_classificacao(y_g, p, inst_g, NOMES, INST_NOMES, '', verbose=False)
        res[nome] = r
        print(f'  {nome:34s} {r["acuracia"]:>7.2f}%' + ''.join(f'{v:>9.1f}%' for v in r['por_classe'])
              + ''.join(f'{r["por_instrumento"][n]["geral"]:>11.1f}%' for n in INST_NOMES))

    print(f'\n  McNemar:')
    comp = {}
    for a in ('D²NN óptica (pof_fixo)', 'D²NN óptica (pof_treinado)'):
        for b in ('filtro |a|', 'D²NN 57 (híbrida)'):
            if b in pred:
                r = mcnemar(y_g, pred[a], pred[b]); comp[f'{a} × {b}'] = r
                print(f'    {a:30s} × {b:20s} Δ = {r["diferenca_pp"]:+6.2f} pp   p = {r["p_valor"]:.2g}')

    rp = res['D²NN óptica (pof_treinado)']
    c1 = rp['acuracia'] >= res['filtro |a|']['acuracia'] - 3.0 and rp['por_classe'][2] >= 95.0
    k57 = 'D²NN óptica (pof_treinado) × D²NN 57 (híbrida)'
    c2 = k57 in comp and comp[k57]['diferenca_pp'] > 0 and comp[k57]['significativo_5pct']
    c3 = abs(rp['acuracia'] - treinos['pof_treinado']['acc_menosV']) < 0.5
    print(f'\n{"="*90}\nCRITÉRIOS\n{"="*90}')
    print(f'  C1 óptica útil (≥ |a| − 3 pp, Mancha ≥ 95%): {rp["acuracia"] - res["filtro |a|"]["acuracia"]:+.2f} pp, '
          f'Mancha {rp["por_classe"][2]:.1f}% → {"ATENDIDO" if c1 else "NÃO ATENDIDO"}')
    if k57 in comp:
        print(f'  C2 melhor que o 57: Δ = {comp[k57]["diferenca_pp"]:+.2f} pp, p = {comp[k57]["p_valor"]:.2g} '
              f'→ {"ATENDIDO" if c2 else "NÃO ATENDIDO"}')
    for nome, info in treinos.items():
        print(f'  C3 polaridade ({nome}): +V {res[f"D²NN óptica ({nome})"]["acuracia"]:.2f}% / '
              f'−V {info["acc_menosV"]:.2f}%, inalteradas {info["inalteradas_pct"]:.1f}%')
    print(f'  → C3 (pof_treinado): {"ATENDIDO" if c3 else "NÃO ATENDIDO"}')
    print(f'\n  Tempo total: {(time.time() - t0) / 60:.1f} min')

    # --- Figura: curvas e fase ---
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for nome, info in treinos.items():
        h = info['historico']
        ax[0].plot(h['val_acc'], label=f'{nome} — val')
    ax[0].axhline(res['filtro |a|']['acuracia'], color='k', ls='--', lw=0.8, label='filtro |a| (teste)')
    ax[0].set_xlabel('época'); ax[0].set_ylabel('acurácia [%]'); ax[0].legend(fontsize=8)
    ax[0].set_title('D²NN puramente óptica — validação')
    im = ax[1].imshow(phi_pof.cpu().numpy(), aspect='auto', cmap='twilight', interpolation='nearest')
    ax[1].set_title('Fase POF (template médio v2), centrada'); fig.colorbar(im, ax=ax[1], label='rad')
    out = FIGURES_DIR / 'cel62_d2nn_optica_pura.png'
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'  Figura: {out.name}')

    save_report('62_d2nn_optica_pura', {
        'janela': list(janela), 'lr': {'fase': LR_FASE, 'leitura': LR_LEITURA}, 'paciencia': PACIENCIA,
        'treinos': treinos, 'resultados': res, 'mcnemar': comp,
        'criterios': {'C1': bool(c1), 'C2': bool(c2), 'C3': bool(c3)},
        'tempo_min': (time.time() - t0) / 60, 'figura': str(out)})


if __name__ == '__main__':
    main()
