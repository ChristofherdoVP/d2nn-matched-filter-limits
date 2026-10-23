"""
train_utils.py — Utilitários de checkpoint e recuperação de sessão para
o treinamento da D²NN: salvar/carregar o modelo, otimizador, histórico
de treino, e os dados pré-processados (X_4f_norm, y_static) em cache
local, evitando reprocessamento desnecessário.

Corresponde à Célula 17 do notebook original (adaptada de Google Drive
para armazenamento local).
"""

import os
from pathlib import Path

import numpy as np
import torch

from Src.config import CHECKPOINTS_DIR, PROCESSED_DIR

# -----------------------------------------------------------------------
# Caminhos padrão (locais, substituindo o Google Drive do notebook)
# -----------------------------------------------------------------------
MODEL_PATH  = CHECKPOINTS_DIR / "modelo_d2nn.pt"
DATA_X_PATH = PROCESSED_DIR / "X_4f_norm.npy"
DATA_Y_PATH = PROCESSED_DIR / "y_static.npy"

# Checkpoint do retreino ideal com entrada observável (script 45)
MODEL_IDEAL_PATH = CHECKPOINTS_DIR / "modelo_d2nn_ideal.pt"

# Checkpoint do treino degradado com entrada observável (script 46).
# (modelo_d2nn_degradado.pt é o do Colab, que não recebeu degradação real)
MODEL_DEG_OBS_PATH = CHECKPOINTS_DIR / "modelo_d2nn_degradado_obs.pt"

# Checkpoint do treino degradado com ruído renovado a cada época (script 47)
MODEL_DEG_AUG_PATH = CHECKPOINTS_DIR / "modelo_d2nn_degradado_aug.pt"

# Checkpoint do treino com campo de referência (detecção homódina, script 48)
MODEL_HOMODINO_PATH = CHECKPOINTS_DIR / "modelo_d2nn_homodino.pt"

# Modelo final congelado (script 49): cópia somente leitura do homódino (48).
# Nunca sobrescrever — uma nova versão ganha um novo nome (v2, v3...).
MODEL_FINAL_PATH = CHECKPOINTS_DIR / "modelo_d2nn_final_v1.pt"

# Experimento de polaridade aleatória (script 52): mesmo protocolo do 48,
# com o sinal de V sorteado a cada época.
#   controle : β congelado em 0 (detecção de intensidade)
#   homódino : β aprendível (campo de referência)
MODEL_POL_CTRL_PATH = CHECKPOINTS_DIR / "modelo_d2nn_pol_controle.pt"
MODEL_POL_HOM_PATH  = CHECKPOINTS_DIR / "modelo_d2nn_pol_homodino.pt"

# Protocolo do 47 com o modelo instrumental corrigido (script 57,
# physics.INSTRUMENTAL_V2; docs/MODELO_INSTRUMENTAL.md).
MODEL_INST_V2_PATH = CHECKPOINTS_DIR / "modelo_d2nn_inst_v2.pt"

# Fator de escala global do treino original: max|X_dc| da receita da
# Célula 25 sobre o dataset (docs/RECONSTRUCAO_NORMALIZACAO.md).
# Usado também na entrada observável, que tem a mesma escala para B ≥ 50 G.
X_SCALE_TREINO = 3.9547e-3

def save_checkpoint(model, optimizer, epoch, loss, history, acc,
                    B3_config, filename=None, extra=None):
    """
    Salva o estado completo do treino (modelo, otimizador, histórico).

    extra : dict opcional com metadados gravados junto (ex.: X_scale,
            receita de entrada, CLASS_DEFS usado) — essenciais para aplicar
            o modelo depois a dados novos.
    """
    if filename is None:
        filename = MODEL_PATH

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'loss': loss,
        'historico': history,
        'melhor_val_acc': acc,
        'epoca_melhor': epoch,
        'B3_config': B3_config,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, filename)
    print(f'➡️ Checkpoint salvo com sucesso em: {filename}')


def load_session(model, model_path=None, data_x_path=None, data_y_path=None):
    """
    Tenta recuperar uma sessão de treino anterior: pesos do modelo,
    histórico de métricas, e dados pré-processados em cache.

    Carrega os pesos do modelo IN-PLACE (strict=False, permite retomar
    mesmo se o classificador mudar de arquitetura depois).

    Retorna
    -------
    dict com: historico, melhor_val_acc, epoca_melhor,
              X_4f_norm (ou None), y_static (ou None), dados_prontos (bool)
    """
    model_path  = Path(model_path)  if model_path  else MODEL_PATH
    data_x_path = Path(data_x_path) if data_x_path else DATA_X_PATH
    data_y_path = Path(data_y_path) if data_y_path else DATA_Y_PATH

    print("🔍 Verificando integridade da sessão anterior...")

    historico = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': [],
    }
    melhor_val_acc = 0.0
    epoca_melhor   = 0

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=next(model.parameters()).device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        resultado_load = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if resultado_load.missing_keys or resultado_load.unexpected_keys:
            print(f'  ⚠️  ATENÇÃO — incompatibilidade ao carregar pesos:')
            if resultado_load.missing_keys:
                print(f'     Chaves ausentes no checkpoint (ficaram com init aleatória):')
                for k in resultado_load.missing_keys:
                    print(f'       - {k}')
            if resultado_load.unexpected_keys:
                print(f'     Chaves no checkpoint sem correspondência no modelo atual:')
                for k in resultado_load.unexpected_keys:
                    print(f'       - {k}')
        else:
            print(f'  ✓ Todos os pesos carregados sem incompatibilidade de shape.')
        
        historico       = checkpoint.get('historico', historico)
        melhor_val_acc  = checkpoint.get('melhor_val_acc', 0.0)
        epoca_melhor    = checkpoint.get('epoca_melhor', 0)
        print(f'✓ Modelo carregado! Retomando da melhor acurácia: {melhor_val_acc:.2f}%')
    else:
        print('⚠ Nenhum modelo prévio encontrado. O treino começará do zero.')

    X_4f_norm = None
    y_static  = None
    dados_prontos = False

    if data_x_path.exists() and data_y_path.exists():
        X_4f_norm = np.load(data_x_path)
        y_static  = np.load(data_y_path)
        print(f'✓ Dados (.npy) carregados. X_4f_norm shape: {X_4f_norm.shape}')
        dados_prontos = True
    else:
        print('⚠ Arquivos .npy não encontrados. Será necessário gerar os dados.')

    return {
        'historico': historico,
        'melhor_val_acc': melhor_val_acc,
        'epoca_melhor': epoca_melhor,
        'X_4f_norm': X_4f_norm,
        'y_static': y_static,
        'dados_prontos': dados_prontos,
    }


# -----------------------------------------------------------------------
# Carregamento do modelo retreinado com degradação instrumental
# Corresponde à Célula 17B do notebook original.
# -----------------------------------------------------------------------

MODEL_DEG_PATH = CHECKPOINTS_DIR / "modelo_d2nn_degradado.pt"


def load_degraded_model(modelo_base, model_deg_path=None, device=None):
    """
    Carrega um segundo modelo (cópia da arquitetura de modelo_base),
    treinado com robustez a degradação instrumental combinada
    (IBIS 40% + CRISP 35% + ViSP 25%, por convenção do treino T2).

    Parâmetros
    ----------
    modelo_base    : instância D2NN já construída (fornece a arquitetura
                     a ser copiada via deepcopy — os pesos são
                     sobrescritos pelo checkpoint carregado)
    model_deg_path : Path ou None (usa MODEL_DEG_PATH por padrão)
    device         : torch.device ou None (usa o device de modelo_base)

    Retorna
    -------
    dict com: modelo_deg (ou None se checkpoint não existir),
              acc_teste, epoca_melhor, val_loss, historico
    """
    import copy

    model_deg_path = Path(model_deg_path) if model_deg_path else MODEL_DEG_PATH
    if device is None:
        device = next(modelo_base.parameters()).device

    print("🔍 Verificando modelo retreinado com degradação...")

    resultado = {
        'modelo_deg': None,
        'acc_teste': 0.0,
        'epoca_melhor': 0,
        'val_loss': 0.0,
        'historico': {},
    }

    if model_deg_path.exists():
        modelo_deg = copy.deepcopy(modelo_base)

        checkpoint_deg = torch.load(model_deg_path, map_location=device)
        modelo_deg.load_state_dict(checkpoint_deg['model_state_dict'], strict=False)
        modelo_deg.to(device)
        modelo_deg.eval()

        acc_deg       = checkpoint_deg.get('acc_teste', 0.0)
        epoca_deg     = checkpoint_deg.get('epoca_melhor', 0)
        val_loss_deg  = checkpoint_deg.get('melhor_val_loss', 0.0)
        historico_deg = checkpoint_deg.get('historico', {})

        print(f'✓ modelo_deg carregado!')
        print(f'  Acurácia teste  : {acc_deg:.2f}%')
        print(f'  Melhor época    : {epoca_deg}')
        print(f'  Melhor val_loss : {val_loss_deg:.5f}')
        print(f'  Instrumentos    : IBIS(40%) + CRISP(35%) + ViSP(25%)')

        resultado.update({
            'modelo_deg': modelo_deg,
            'acc_teste': acc_deg,
            'epoca_melhor': epoca_deg,
            'val_loss': val_loss_deg,
            'historico': historico_deg,
        })
    else:
        print(f'⚠ {model_deg_path.name} não encontrado.')
        print('  Rode o treino de degradação (T2) primeiro para gerar o checkpoint.')

    return resultado

from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

from Src.config import DTYPE


def split_indices(y, B3, seed):
    """
    Split estratificado treino/validação/teste — MESMA lógica e mesma
    semente de prepare_training_tensors (índices idênticos).
    Retorna (idx_train, idx_val, idx_test).
    """
    zeros = np.zeros(len(y))
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=B3['test_frac'],
                                      random_state=seed)
    idx_trainval, idx_test = next(sss_test.split(zeros, y))
    val_frac_adj = B3['val_frac'] / (1 - B3['test_frac'])
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_adj,
                                     random_state=seed)
    idx_train, idx_val = next(sss_val.split(zeros[idx_trainval], y[idx_trainval]))
    idx_train, idx_val = idx_trainval[idx_train], idx_trainval[idx_val]

    # Integridade (origem: Célula 24, Teste 1): conjuntos disjuntos e cobertura total.
    # Observação: isto não detecta vazamento pela ENTRADA (ex.: a receita antiga
    # do Colab codificava B e f verdadeiros) — só vazamento de índices.
    s_tr, s_va, s_te = set(idx_train.tolist()), set(idx_val.tolist()), set(idx_test.tolist())
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), \
        'split_indices: índices compartilhados entre treino, validação e teste'
    assert len(s_tr | s_va | s_te) == len(y), 'split_indices: o split não cobre o dataset'
    return idx_train, idx_val, idx_test


def prepare_training_tensors(X_static, y_static, B_all, F_all,
                              B_norm_all, F_norm_all, boundary_all,
                              modelo, B3, seed, device=None,
                              boundary_weight=0.5, X_4f_norm=None,
                              entrada='observavel', perfis=None,
                              X_scale='treino', verbose=True):
    """
    Prepara os DataLoaders de treino/validação/teste.

    Se X_4f_norm não for fornecido, monta os espectrogramas conforme `entrada`:
      'observavel' (padrão) → só a partir dos perfis (I, V):
                              Src.optics.build_observable_spectrograms.
                              `perfis=(I, V)`; se None, usa perfis ideais
                              (sem ruído) de Src.dataset.generate_ideal_profiles.
      'colab'               → receita ORIGINAL do Colab, que depende de B e f
                              (Src.optics.build_sweep_spectrograms). Apenas
                              para reproduzir o checkpoint antigo.

    Em seguida divide pelo fator de escala:
      X_scale='treino' → X_SCALE_TREINO
      X_scale='auto'   → máximo absoluto dos dados gerados
      X_scale=<float>  → valor explícito

    Retorna dict com: train_loader, val_loader, test_loader, X_4f_norm,
    X_scale (None se os dados vieram prontos), idx_train, idx_val, idx_test
    """
    from Src.optics import build_sweep_spectrograms, build_observable_spectrograms
    from Src.dataset import generate_ideal_profiles

    if device is None:
        device = next(modelo.parameters()).device

    X_scale_usado = None
    if X_4f_norm is not None:
        print('✓ X_4f_norm já disponível — pulando geração.')
    else:
        if entrada == 'colab':
            print('Gerando espectrogramas (receita ORIGINAL do Colab — depende de B e f)...')
            X_4f_norm = build_sweep_spectrograms(B_all, F_all, verbose=verbose)
        elif entrada == 'observavel':
            if perfis is None:
                print('Gerando perfis ideais (sem ruído)...')
                perfis = generate_ideal_profiles(B_all, F_all)
            print('Montando espectrogramas (entrada observável: só perfis I, V)...')
            X_4f_norm = build_observable_spectrograms(perfis[0], perfis[1])
        else:
            raise ValueError(f"entrada deve ser 'observavel' ou 'colab', recebido: {entrada}")

        std_linhas = float(X_4f_norm[::50].std(axis=1).mean())
        assert std_linhas > 0.0, 'ERRO: linhas idênticas — receita de montagem incorreta'

        if X_scale == 'treino':
            X_scale_usado = X_SCALE_TREINO
        elif X_scale == 'auto':
            X_scale_usado = float(np.abs(X_4f_norm).max()) + 1e-10
        else:
            X_scale_usado = float(X_scale)
        X_4f_norm /= X_scale_usado
        print(f'  X_scale = {X_scale_usado:.5e}  |  '
              f'faixa: [{X_4f_norm.min():.4f}, {X_4f_norm.max():.4f}]')

    # --- Tensores com targets de regressão e pesos de boundary ---
    sample_weights = np.where(boundary_all, boundary_weight, 1.0).astype(np.float32)

    X_tensor      = torch.tensor(X_4f_norm,     dtype=DTYPE,      device=device)
    y_tensor      = torch.tensor(y_static,       dtype=torch.long, device=device)
    B_norm_tensor = torch.tensor(B_norm_all,     dtype=DTYPE,      device=device)
    F_norm_tensor = torch.tensor(F_norm_all,     dtype=DTYPE,      device=device)
    w_tensor      = torch.tensor(sample_weights, dtype=DTYPE,      device=device)

    if verbose:
        print(f'\n  X_tensor      : {tuple(X_tensor.shape)}')
        print(f'  B_norm_tensor : {tuple(B_norm_tensor.shape)}  (target regressão B)')
        print(f'  F_norm_tensor : {tuple(F_norm_tensor.shape)}  (target regressão f)')
        print(f'  w_tensor      : boundary={boundary_weight}, normal=1.0  '
              f'({boundary_all.sum()} amostras boundary)')

    # --- Split estratificado ---
    y_np = y_static
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=B3['test_frac'],
                                      random_state=seed)
    idx_trainval, idx_test = next(sss_test.split(X_4f_norm, y_np))

    val_frac_adj = B3['val_frac'] / (1 - B3['test_frac'])
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_adj,
                                     random_state=seed)
    idx_train, idx_val = next(sss_val.split(X_4f_norm[idx_trainval],
                                             y_np[idx_trainval]))
    idx_train = idx_trainval[idx_train]
    idx_val   = idx_trainval[idx_val]

    # --- DataLoaders: 5 elementos por batch (X, y, B_norm, F_norm, weight) ---
    dataset_full = TensorDataset(X_tensor, y_tensor,
                                  B_norm_tensor, F_norm_tensor, w_tensor)

    train_loader = DataLoader(Subset(dataset_full, idx_train),
                              batch_size=B3['batch_size'],
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(Subset(dataset_full, idx_val),
                              batch_size=B3['batch_size'], shuffle=False)
    test_loader  = DataLoader(Subset(dataset_full, idx_test),
                              batch_size=B3['batch_size'], shuffle=False)

    if verbose:
        print(f'\n{"="*55}')
        print('SPLIT DO DATASET')
        print(f'{"="*55}')
        print(f'  {"Conj":10s}  {"N":>6s}  {"Quieta":>6s}  '
              f'{"Rede":>6s}  {"Mancha":>6s}  {"Bound":>5s}')
        print('-' * 55)
        for nome, idx in [('Train', idx_train), ('Val', idx_val), ('Test', idx_test)]:
            ys = y_np[idx]
            bs = boundary_all[idx]
            print(f'  {nome:10s}  {len(idx):>6d}  {(ys==0).sum():>6d}  '
                  f'{(ys==1).sum():>6d}  {(ys==2).sum():>6d}  {bs.sum():>5d}')
        print(f'{"="*55}')

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'X_4f_norm': X_4f_norm,
        'X_scale': X_scale_usado,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test,
    }


if __name__ == "__main__":
    from Src.config import DEVICE
    from Src.model import D2NN, PUPIL_T, PHI_VORTICE

    modelo = D2NN(pupila=PUPIL_T, phi_vortice=PHI_VORTICE).to(DEVICE)
    session = load_session(modelo)

    print(f'\n✓ Módulo train_utils.py testado.')
    print(f'  dados_prontos  : {session["dados_prontos"]}')
    print(f'  melhor_val_acc : {session["melhor_val_acc"]}')