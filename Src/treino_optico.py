"""
treino_optico.py — Treino da D²NN puramente óptica (D2NNOptica): fase POF inicial e laço
de treino próprio (o train_model é específico da D2NN híbrida).

Extraído do script 62 sem mudar a lógica (scripts 62 e 67).
  dados_v2             : dataset principal, split, perfis v2 com ruído fixo e renovado
  fase_pof             : φ = −arg F{X_template} do template médio v2 (com diafragma opcional)
  mascara_casada       : (φ, a) de H = S*/max|S| do template médio v2 (script 74)
  treinar              : Adam (fase e leitura), parada antecipada em val loss
  avaliar              : previsões na ordem do loader
"""

import numpy as np
import torch
import torch.nn.functional as Fnn
from torch.utils.data import TensorDataset, DataLoader, Subset

from Src.config import DEVICE, DTYPE, SEED
from Src.model import B3, PUPIL_T, D2NNOptica, D2NNOpticaComplexa
from Src.physics import INSTRUMENTAL_V2, LAM, CALIB
from Src.dataset import (generate_master_dataset, generate_degraded_profiles,
                         INST_NOMES, INST_PROBS, NOISE_STD, SEED_BASE)
from Src.train_utils import split_indices
from Src.evaluation import templates_filtro_casado

INSTR             = INSTRUMENTAL_V2
BOUNDARY_WEIGHT   = 0.5
PESOS_CLASSE      = [1.5, 1.5, 1.0]
SEED_DEG_FIXO     = SEED + 2
SEED_DEG_EPOCA    = SEED + 10_000
LR_FASE, LR_LEITURA = 1e-2, 1e-2
PACIENCIA         = 15


def dados_v2():
    """Protocolo do 57/62. Devolve dict com loaders, renovar(), V_fixo, inst_fixo, y, idx."""
    (_, y, B_all, F_all, B_norm, F_norm, boundary) = generate_master_dataset(NOISE_STD, SEED_BASE, 'static')
    _, V_fixo, inst_fixo = generate_degraded_profiles(B_all, F_all, seed=SEED_DEG_FIXO, instrumental=INSTR)
    idx_train, idx_val, idx_test = split_indices(y, B3, SEED)
    w = np.where(boundary, BOUNDARY_WEIGHT, 1.0).astype(np.float32)
    V_t = torch.tensor(V_fixo, dtype=DTYPE, device=DEVICE)
    ds = TensorDataset(V_t, torch.tensor(y, dtype=torch.long, device=DEVICE),
                       torch.tensor(B_norm, dtype=DTYPE, device=DEVICE),
                       torch.tensor(F_norm, dtype=DTYPE, device=DEVICE),
                       torch.tensor(w, dtype=DTYPE, device=DEVICE))
    train_loader = DataLoader(Subset(ds, idx_train), batch_size=B3['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(Subset(ds, idx_val), batch_size=B3['batch_size'])
    idx_train_t = torch.tensor(idx_train, device=DEVICE)

    def renovar(ep):
        _, V_novo, _ = generate_degraded_profiles(B_all[idx_train], F_all[idx_train],
                                                  seed=SEED_DEG_EPOCA + ep, instrumental=INSTR)
        V_t[idx_train_t] = torch.tensor(V_novo, dtype=DTYPE, device=DEVICE)

    return {'train_loader': train_loader, 'val_loader': val_loader, 'renovar': renovar,
            'V_fixo': V_fixo, 'inst_fixo': inst_fixo, 'y': y, 'idx_train': idx_train}


def janela_diafragma(diafragma_pm):
    if diafragma_pm is None:
        return np.ones_like(LAM)
    return (np.abs((LAM - CALIB['lambda0_core']) * 1e3) <= float(diafragma_pm)).astype(np.float64)


def fase_pof(montar, diafragma_pm=None):
    """φ = −arg F{X_template}, template médio v2 ponderado por INST_PROBS e multiplicado pelo diafragma."""
    templ = templates_filtro_casado(INST_NOMES, instrumental=INSTR)
    p = np.asarray([INST_PROBS[n] for n in INST_NOMES] if isinstance(INST_PROBS, dict) else INST_PROBS, dtype=float)
    t = sum(p[k] * templ[k] for k in range(len(INST_NOMES))) / p.sum()
    t = t * janela_diafragma(diafragma_pm)
    Xt = montar(torch.tensor(t[None, :], dtype=DTYPE, device=DEVICE))[0]
    S = torch.fft.fft2(Xt.to(torch.complex64), norm='ortho')
    return torch.fft.fftshift(-torch.angle(S), dim=(-2, -1)).to(DTYPE)


def mascara_casada(montar, diafragma_pm=None):
    """(φ, a) da máscara casada complexa H = S*/max|S|, template médio v2 (como fase_pof)."""
    templ = templates_filtro_casado(INST_NOMES, instrumental=INSTR)
    p = np.asarray([INST_PROBS[n] for n in INST_NOMES] if isinstance(INST_PROBS, dict) else INST_PROBS, dtype=float)
    t = sum(p[k] * templ[k] for k in range(len(INST_NOMES))) / p.sum()
    t = t * janela_diafragma(diafragma_pm)
    Xt = montar(torch.tensor(t[None, :], dtype=DTYPE, device=DEVICE))[0]
    S = torch.fft.fft2(Xt.to(torch.complex64), norm='ortho')
    fase = torch.fft.fftshift(-torch.angle(S), dim=(-2, -1)).to(DTYPE)
    amp = torch.fft.fftshift(S.abs() / S.abs().max(), dim=(-2, -1)).to(DTYPE)
    return fase, amp


def avaliar(modelo, loader, montar):
    modelo.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            out.append(modelo(montar(batch[0])).argmax(1).cpu().numpy())
    return np.concatenate(out)


def _perda(modelo, montar, batch, pesos):
    V, y, _, _, w = batch
    logits = modelo(montar(V))
    return (Fnn.cross_entropy(logits, y, weight=pesos, reduction='none') * w).mean(), logits, y


def treinar(nome, fase_init, treinar_fase, janela, montar, dados, max_epocas, diafragma_pm=None, amp_init=None):
    """amp_init=None → D2NNOptica (só fase); amp_init dado → D2NNOpticaComplexa (amplitude e fase)."""
    train_loader, val_loader, renovar = dados['train_loader'], dados['val_loader'], dados['renovar']
    if amp_init is None:
        modelo = D2NNOptica(fase_init, janela, treinar_fase, diafragma_pm=diafragma_pm).to(DEVICE)
    else:
        modelo = D2NNOpticaComplexa(fase_init, amp_init, janela, treinar_fase, diafragma_pm=diafragma_pm).to(DEVICE)
    with torch.no_grad():
        z = torch.cat([torch.log(modelo.energia(montar(b[0])) + 1e-12) for b in train_loader])
        modelo.mu.fill_(float(z.mean())); modelo.s.fill_(float(z.std()) or 1.0)
    grupos = [{'params': [modelo.w, modelo.b], 'lr': LR_LEITURA}]
    if treinar_fase:
        grupos.append({'params': [p for p in modelo.mascara.parameters() if p.requires_grad], 'lr': LR_FASE})
    opt = torch.optim.Adam(grupos)
    pesos = torch.tensor(PESOS_CLASSE, dtype=DTYPE, device=DEVICE)
    melhor, estado, epoca_melhor, sem = float('inf'), None, 0, 0
    hist = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    print(f'\n  Treino — {nome}  (fase {"treinável" if treinar_fase else "fixa"}, janela {janela}, '
          f'diafragma {"±" + str(diafragma_pm) + " pm" if diafragma_pm else "nenhum"})')
    for ep in range(1, max_epocas + 1):
        renovar(ep)
        modelo.train()
        tl = []
        for batch in train_loader:
            opt.zero_grad()
            loss, _, _ = _perda(modelo, montar, batch, pesos)
            loss.backward(); opt.step(); tl.append(loss.item())
        modelo.eval()
        vl, ac, n = [], 0, 0
        with torch.no_grad():
            for batch in val_loader:
                loss, logits, y = _perda(modelo, montar, batch, pesos)
                vl.append(loss.item() * len(y)); ac += int((logits.argmax(1) == y).sum()); n += len(y)
        vloss, vacc = sum(vl) / n, 100 * ac / n
        hist['train_loss'].append(float(np.mean(tl))); hist['val_loss'].append(vloss); hist['val_acc'].append(vacc)
        marca = ''
        if vloss < melhor:
            melhor, epoca_melhor, sem, marca = vloss, ep, 0, ' ★'
            estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sem += 1
        if ep == 1 or ep % 10 == 0:
            print(f'    época {ep:3d}  loss T {np.mean(tl):.4f}  loss V {vloss:.4f}  acc V {vacc:5.1f}%{marca}')
        if sem >= PACIENCIA:
            print(f'    parada antecipada na época {ep}')
            break
    modelo.load_state_dict(estado)
    dphi = modelo.mascara.delta_phi.detach()
    pup = PUPIL_T.abs() > 0
    print(f'    melhor época {epoca_melhor} (val loss {melhor:.4f}, val acc {hist["val_acc"][epoca_melhor - 1]:.1f}%); '
          f'std(Δφ) = {float(dphi[pup].std()):.3e} rad')
    return modelo, {'epoca_melhor': epoca_melhor, 'melhor_val_loss': melhor, 'historico': hist,
                    'std_dphi': float(dphi[pup].std())}
