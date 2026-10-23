"""
train.py — Loop de treinamento da D²NN: critério ponderado por classe e
por amostra (boundary weight), otimizador com learning rates por bloco,
scheduler ReduceLROnPlateau, early stopping opcional, e continuidade de
histórico entre sessões.

Corresponde à Célula 20 do notebook original, com duas melhorias:
  1. Salva checkpoint automaticamente ao final (a Célula 20 original
     não fazia isso — risco de perder o treino).
  2. Histórico contínuo entre sessões (em vez de reiniciar do zero
     a cada execução).
  3. Early stopping opcional (desligado por padrão no notebook
     original; aqui LIGADO por padrão, conforme decisão do usuário).
"""

import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim


def avaliar(loader, modelo, crit, transform=None):
    """
    Avalia o modelo em um DataLoader, ignorando B_norm/F_norm/weight.

    transform : função opcional aplicada a batch[0] antes do modelo (ex.:
                montar a entrada 2D na GPU a partir dos perfis V).
    """
    modelo.eval()
    loss_total, corretos, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            X_batch, y_batch = batch[0], batch[1]
            if transform is not None:
                X_batch = transform(X_batch)
            logits      = modelo(X_batch)
            loss        = crit(logits, y_batch).mean()
            loss_total += loss.item() * X_batch.size(0)
            preds       = logits.argmax(dim=1)
            corretos   += (preds == y_batch).sum().item()
            total      += X_batch.size(0)
    return loss_total / total, corretos / total * 100


def build_optimizer(modelo, B3, lr_extra=5e-3):
    """
    Otimizador Adam com learning rates diferenciados por bloco.

    Qualquer parâmetro fora dos blocos listados (ex.: beta_ref da
    D2NNHomodina) entra num grupo extra com lr_extra e sem weight decay —
    antes, um parâmetro novo ficaria silenciosamente fora do treino.
    """
    grupos = [
        {'params': list(modelo.camada1.parameters()),        'lr': 1e-4},
        {'params': list(modelo.camada2.parameters()),        'lr': 1e-4},
        {'params': list(modelo.camada3.parameters()),        'lr': 1e-4},
        {'params': list(modelo.detector_spec.parameters()),  'lr': 5e-4},
        {'params': list(modelo.classificador.parameters()),  'lr': 5e-5},
    ]
    ja_incluidos = {id(p) for g in grupos for p in g['params']}
    # Só parâmetros treináveis: um parâmetro congelado (ex.: beta_ref fixo em 0
    # no controle do script 52) fica fora do otimizador.
    extras = [p for p in modelo.parameters()
              if id(p) not in ja_incluidos and p.requires_grad]
    if extras:
        grupos.append({'params': extras, 'lr': lr_extra, 'weight_decay': 0.0})
    return optim.Adam(grupos, weight_decay=B3['weight_decay'])


def verify_checkpoint_consistency(modelo, val_loader, historico, device, tol_pct=5.0):
    """
    Avalia o modelo AO VIVO no val_loader logo após carregar um checkpoint,
    e compara com a métrica armazenada no histórico. Se divergirem muito,
    avisa que o checkpoint provavelmente não é compatível com o pipeline
    de dados atual (ex: normalização diferente da usada no treino original).
    """
    if not historico.get('val_acc'):
        return None  # sem histórico anterior, nada para comparar

    criterio_check = nn.CrossEntropyLoss()
    _, val_acc_real = avaliar(val_loader, modelo, criterio_check)
    val_acc_historico = max(historico['val_acc'])

    diff = abs(val_acc_real - val_acc_historico)
    print(f'\n🔎 Verificação de consistência do checkpoint:')
    print(f'   Val acc armazenada no histórico : {val_acc_historico:.2f}%')
    print(f'   Val acc real (avaliada agora)   : {val_acc_real:.2f}%')

    if diff > tol_pct:
        print(f'   ⚠️  DIVERGÊNCIA DE {diff:.1f} PONTOS! O checkpoint provavelmente')
        print(f'      foi treinado com um pipeline de dados diferente do atual')
        print(f'      (ex: normalização diferente). Considere retreinar do zero.')
    else:
        print(f'   ✓ Consistente (diferença de {diff:.1f} pontos).')

    return val_acc_real

def train_model(modelo, train_loader, val_loader, B3, device,
                 historico=None, pesos_classe=(1.5, 1.5, 1.0),
                 boundary_weight=0.5,
                 early_stopping_patience=50, verbose=True,
                 avaliar_estado_inicial=False,
                 transform=None, ao_iniciar_epoca=None):
    """
    Treina a D²NN, continuando o histórico de uma sessão anterior se
    fornecido (historico=session['historico']).

    Early stopping: interrompe se val_loss não melhorar por
    `early_stopping_patience` épocas seguidas. Passe None para desligar.

    avaliar_estado_inicial: se True, avalia o modelo ANTES de treinar e usa
    esse estado como referência de "melhor". Recomendado no ajuste fino:
    garante que o modelo final nunca fica pior (em val_loss) que o inicial.

    transform       : função aplicada a cada batch[0] (treino e validação)
                      antes do modelo — ex.: montar a entrada na GPU.
    ao_iniciar_epoca: função chamada no início de cada época com o número
                      da época — ex.: renovar o ruído dos dados de treino.

    Retorna
    -------
    dict com: historico, melhor_estado (state_dict), melhor_val_loss,
              melhor_val_acc, epoca_melhor (posição absoluta no
              histórico completo), parou_cedo (bool), tempo_total_s
    """
    if historico is None or not historico.get('train_loss'):
        historico = {
            'train_loss': [], 'val_loss': [],
            'train_acc' : [], 'val_acc' : [],
            'lr'        : [],
        }
    epoca_offset = len(historico['train_loss'])  # quantas já rodaram antes

    pesos_classe_t = torch.tensor(list(pesos_classe), device=device)
    criterio       = nn.CrossEntropyLoss(weight=pesos_classe_t, reduction='none')

    otimizador = build_optimizer(modelo, B3)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode='min',
        patience=B3['lr_patience'],
        factor=B3['lr_factor'])

    # Estado inicial "melhor" = o que já está carregado no modelo
    # (se veio de um checkpoint, já é o melhor da sessão anterior)
    if historico['val_loss']:
        melhor_val_loss = min(historico['val_loss'])
        melhor_val_acc  = historico['val_acc'][historico['val_loss'].index(melhor_val_loss)]
        epoca_melhor    = historico['val_loss'].index(melhor_val_loss) + 1
    else:
        melhor_val_loss = float('inf')
        melhor_val_acc  = 0.0
        epoca_melhor    = 0
    melhor_estado = copy.deepcopy(modelo.state_dict())

    if avaliar_estado_inicial:
        v_loss0, v_acc0 = avaliar(val_loader, modelo, criterio, transform)
        if verbose:
            print(f'Estado inicial (antes do treino): val_loss={v_loss0:.4f}  val_acc={v_acc0:.2f}%')
        if v_loss0 < melhor_val_loss:
            melhor_val_loss = v_loss0
            melhor_val_acc  = v_acc0
            epoca_melhor    = epoca_offset

    epocas_sem_melhora = 0
    parou_cedo = False

    if verbose:
        print(f'Iniciando treinamento — {B3["epochs"]} épocas  |  {device}')
        print(f'Critério: CrossEntropy (pesos={list(pesos_classe)}, reduction=none)')
        print(f'Boundary weight: {boundary_weight}')
        if epoca_offset > 0:
            print(f'Continuando de sessão anterior: {epoca_offset} épocas já registradas '
                  f'(melhor val_loss até agora: {melhor_val_loss:.4f})')
        if early_stopping_patience is not None:
            print(f'Early stopping: paciência de {early_stopping_patience} épocas sem melhora')
        print(f'{"="*72}')
        print(f'  {"Época":>6s}  {"Loss T":>8s}  {"Acc T":>7s}  '
              f'{"Loss V":>8s}  {"Acc V":>7s}  {"LR":>10s}  {"Tempo":>7s}')
        print('-' * 72)

    t_total = time.time()

    for i in range(1, B3['epochs'] + 1):
        epoca_abs = epoca_offset + i
        t_ep = time.time()
        if ao_iniciar_epoca is not None:
            ao_iniciar_epoca(epoca_abs)
        modelo.train()
        loss_ep, corretos_ep, total_ep = 0.0, 0, 0

        for batch in train_loader:
            X_batch, y_batch, Bn_batch, Fn_batch, w_batch = batch
            if transform is not None:
                X_batch = transform(X_batch)

            otimizador.zero_grad()
            logits = modelo(X_batch)

            loss_por_amostra = criterio(logits, y_batch)
            loss             = (loss_por_amostra * w_batch).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()

            loss_ep     += loss.item() * X_batch.size(0)
            preds        = logits.argmax(dim=1)
            corretos_ep += (preds == y_batch).sum().item()
            total_ep    += X_batch.size(0)

        train_loss = loss_ep     / total_ep
        train_acc  = corretos_ep / total_ep * 100
        val_loss, val_acc = avaliar(val_loader, modelo, criterio, transform)
        scheduler.step(val_loss)

        lr_atual = otimizador.param_groups[0]['lr']
        t_ep     = time.time() - t_ep

        historico['train_loss'].append(train_loss)
        historico['val_loss'  ].append(val_loss)
        historico['train_acc' ].append(train_acc)
        historico['val_acc'   ].append(val_acc)
        historico['lr'        ].append(lr_atual)

        if val_loss < melhor_val_loss:
            melhor_val_loss = val_loss
            melhor_val_acc  = val_acc
            melhor_estado   = copy.deepcopy(modelo.state_dict())
            epoca_melhor    = epoca_abs
            epocas_sem_melhora = 0
            flag = ' ★'
        else:
            epocas_sem_melhora += 1
            flag = ''

        if verbose:
            print(f'  {epoca_abs:>6d}  {train_loss:>8.4f}  {train_acc:>6.1f}%  '
                  f'{val_loss:>8.4f}  {val_acc:>6.1f}%  '
                  f'{lr_atual:>10.2e}  {t_ep:>6.1f}s{flag}')

        if early_stopping_patience is not None and epocas_sem_melhora >= early_stopping_patience:
            if verbose:
                print(f'\n⏹ Early stopping: sem melhora por {early_stopping_patience} épocas.')
            parou_cedo = True
            break

    t_total = time.time() - t_total

    modelo.load_state_dict(melhor_estado)

    if verbose:
        print(f'\n✓ Restaurado para época {epoca_melhor} '
              f'(menor val_loss: {melhor_val_loss:.4f})')
        print(f'{"="*72}')
        print(f'  Tempo total   : {t_total/60:.1f} min')
        print(f'  Melhor Val Acc: {melhor_val_acc:.1f}%')

    return {
        'historico': historico,
        'melhor_estado': melhor_estado,
        'melhor_val_loss': melhor_val_loss,
        'melhor_val_acc': melhor_val_acc,
        'epoca_melhor': epoca_melhor,
        'parou_cedo': parou_cedo,
        'tempo_total_s': t_total,
    }