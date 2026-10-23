"""
dataset.py — Geração do Dataset Master (perfis Stokes V rotulados) para
treino da D²NN.

Classes: Região Quieta (0), Rede Magnética (1), Mancha Solar (2).
As fronteiras entre classes são contíguas (sem gap), corrigido para
evitar rótulos contraditórios na zona de transição B/f.

A versão anterior à correção (CLASS_DEFS_COLAB_ORIGINAL, com gap em
f ∈ (0.20, 0.30)) é mantida apenas para reproduzir os dados com que o
checkpoint original do Colab foi treinado (ver scripts 38–44).

Corresponde à Célula 4 do notebook original.
"""

import numpy as np

from Src.physics import (get_stokes, CALIB, V_CORE_KMS, LAM, INSTRUMENTAL, degradar_stokes,
                         reamostrar_para_LAM)

# -----------------------------------------------------------------------
# Parâmetros do experimento
# -----------------------------------------------------------------------
N_SAMPLES  = 500       # amostras por classe → total 1500
NOISE_STD  = 1e-3      # ruído branco [Ic]
SEED_BASE  = 42

B_MAX_NORM = 2500.0    # normalização para regressão
F_MAX_NORM = 1.0

# Limites da zona de transição (boundary flag)
B_TRANSICAO_MIN = 450.0
B_TRANSICAO_MAX = 550.0
F_TRANSICAO_MIN = 0.25
F_TRANSICAO_MAX = 0.35

# -----------------------------------------------------------------------
# CLASS_DEFS — fronteiras contíguas, sem gap (versão corrigida, PADRÃO)
# -----------------------------------------------------------------------
CLASS_DEFS = [
    (0,    100,  0.00, 0.05, 0, 'Região Quieta'),
    (100,  500,  0.05, 0.30, 1, 'Rede Magnética'),
    (500, 2500,  0.30, 1.00, 2, 'Mancha Solar'),
]

# Verificação de integridade: fronteiras contíguas, sem gap
_f_maxs = [d[3] for d in CLASS_DEFS[:-1]]
_f_mins = [d[2] for d in CLASS_DEFS[1:]]
for _i, (_fmax, _fmin) in enumerate(zip(_f_maxs, _f_mins)):
    assert abs(_fmax - _fmin) < 1e-9, \
        f"GAP entre classes {_i} e {_i+1}: f_max={_fmax} ≠ f_min={_fmin}"

# Versão ANTERIOR à correção do gap, usada para gerar o X_4f_norm do Colab
# com que o checkpoint de 97,35% foi treinado. Tem gap em f ∈ (0.20, 0.30).
# Mantida apenas para reprodutibilidade (ver scripts 38–44).
# Fica DEPOIS do assert de propósito: o assert só verifica o CLASS_DEFS padrão.
CLASS_DEFS_COLAB_ORIGINAL = [
    (0,    100,  0.00, 0.05, 0, 'Região Quieta'),
    (100,  500,  0.05, 0.20, 1, 'Rede Magnética'),
    (500, 2500,  0.30, 1.00, 2, 'Mancha Solar'),
]


# -----------------------------------------------------------------------
# Funções auxiliares
# -----------------------------------------------------------------------
def _is_boundary(b, f):
    return bool((B_TRANSICAO_MIN <= b <= B_TRANSICAO_MAX) or
                (F_TRANSICAO_MIN <= f <= F_TRANSICAO_MAX))


def generate_stokes_V_sample(B, filling, v_km_s=0.0, rng=None, noise=NOISE_STD):
    """Mantida para compatibilidade com código existente."""
    st = get_stokes(B, theta_deg=0.0, filling=filling, v_km_s=v_km_s)
    noise_arr = (rng.normal(0, noise, CALIB['n_lam'])
                 if rng is not None else np.zeros(CALIB['n_lam']))
    return (st['V'] + noise_arr).astype(np.float32)


def generate_master_dataset(noise, seed, mode='static', n_samples=N_SAMPLES,
                            class_defs=None):
    """
    Gera o Dataset Master.

    Parâmetros
    ----------
    noise      : desvio padrão do ruído branco [Ic]
    seed       : semente do gerador aleatório
    mode       : 'static', 'doppler' ou 'gradient'
    n_samples  : amostras por classe
    class_defs : definição das classes; None → CLASS_DEFS (corrigido, padrão).
                 Use CLASS_DEFS_COLAB_ORIGINAL para reproduzir os dados do
                 treino original do Colab.

    Retorna: X, y, B_arr, F_arr, B_norm, F_norm, boundary
    """
    rng = np.random.default_rng(seed)
    X, y            = [], []
    B_list, F_list  = [], []
    B_norm_list     = []
    F_norm_list     = []
    boundary_list   = []

    for b_min, b_max, f_min, f_max, lid, _ in (class_defs or CLASS_DEFS):
        for _ in range(n_samples):
            b = rng.uniform(b_min, b_max)
            f = rng.uniform(f_min, f_max)

            if mode == 'static':
                sig_v = get_stokes(b, filling=f, v_km_s=0.0)['V']
            elif mode == 'doppler':
                v     = rng.uniform(-5.0, 5.0)
                sig_v = get_stokes(b, filling=f, v_km_s=v)['V']
            else:   # gradient
                v_bulk = rng.uniform(-5.0, 5.0)
                v_grad = rng.normal(0.0, 0.5)
                st_lo  = get_stokes(b, filling=f, v_km_s=v_bulk - v_grad/2)['V']
                st_hi  = get_stokes(b, filling=f, v_km_s=v_bulk + v_grad/2)['V']
                sig_v  = 0.5 * (st_lo + st_hi)

            noise_arr = rng.normal(0, noise, CALIB['n_lam'])
            X.append((sig_v + noise_arr).astype(np.float32))
            y.append(lid)
            B_list.append(b)
            F_list.append(f)
            B_norm_list.append(b / B_MAX_NORM)
            F_norm_list.append(f / F_MAX_NORM)
            boundary_list.append(_is_boundary(b, f))

    return (
        np.array(X,             dtype=np.float32),
        np.array(y,             dtype=np.int64),
        np.array(B_list,        dtype=np.float32),
        np.array(F_list,        dtype=np.float32),
        np.array(B_norm_list,   dtype=np.float32),
        np.array(F_norm_list,   dtype=np.float32),
        np.array(boundary_list, dtype=bool),
    )


def generate_ideal_profiles(B_arr, F_arr, v_km_s=V_CORE_KMS):
    """
    Perfis de Stokes I e V SEM ruído para cada amostra — a "observação
    ideal". Usados no treino ideal com a entrada observável
    (Src.optics.build_observable_spectrograms).

    Retorna (I, V), cada um (N, N_LAM) em float64.
    """
    N = len(B_arr)
    I = np.empty((N, CALIB['n_lam']), dtype=np.float64)
    V = np.empty((N, CALIB['n_lam']), dtype=np.float64)
    for i in range(N):
        st = get_stokes(float(B_arr[i]), filling=float(F_arr[i]), v_km_s=v_km_s)
        I[i], V[i] = st['I'], st['V']
    return I, V


# Instrumentos e probabilidades de sorteio do treino degradado (como na Célula 20B)
INST_NOMES = ['IBIS/DST', 'CRISP/SST', 'ViSP/DKIST']
INST_PROBS = [0.40, 0.35, 0.25]


def generate_degraded_profiles(B_arr, F_arr, seed, inst_probs=None,
                               v_km_s=V_CORE_KMS, fator_ruido=1.0, instrumental=None):
    """
    Perfis de Stokes I e V DEGRADADOS por um instrumento sorteado por amostra.

    Para cada amostra: perfil puro → degradar_stokes() com os parâmetros do
    instrumento (luz espúria → LSF → reamostragem na grade do instrumento →
    ruído fotônico + leitura) → interpolação linear de volta para a grade
    LAM (3000 pontos), que é a grade de entrada do modelo.

    Diferente da Célula 20B original, aqui a degradação é EFETIVAMENTE usada.

    instrumental : dicionário de instrumentos. None = INSTRUMENTAL (modelo LEGADO,
                   usado nos scripts 46–54; resultados idênticos aos anteriores).
                   Para o modelo corrigido: physics.INSTRUMENTAL_V2 (docs/MODELO_INSTRUMENTAL.md).
    fator_ruido : multiplica o ruído (1 = instrumentos reais; 0 = sem ruído, com luz
                  espúria, LSF e amostragem mantidas). Não altera a sequência
                  aleatória: mesma semente → mesmos instrumentos e mesma
                  realização de ruído, só escalada.

    Retorna (I, V, inst_idx):
      I, V     : (N, N_LAM) float64 na grade LAM
      inst_idx : (N,) int — índice em INST_NOMES do instrumento de cada amostra
    """
    if inst_probs is None:
        inst_probs = INST_PROBS
    if instrumental is None:
        instrumental = INSTRUMENTAL
    rng = np.random.default_rng(seed)
    N = len(B_arr)
    I = np.empty((N, CALIB['n_lam']), dtype=np.float64)
    V = np.empty((N, CALIB['n_lam']), dtype=np.float64)
    inst_idx = np.empty(N, dtype=np.int64)

    for i in range(N):
        k = int(rng.choice(len(INST_NOMES), p=inst_probs))
        inst_idx[i] = k
        st  = get_stokes(float(B_arr[i]), filling=float(F_arr[i]), v_km_s=v_km_s)
        instr = instrumental[INST_NOMES[k]]
        deg = degradar_stokes(st, LAM, instr, rng_local=rng, fator_ruido=fator_ruido)
        I[i], V[i] = reamostrar_para_LAM(deg, instr)

    return I, V, inst_idx


def print_master_report(title, X, y, B, F, boundary, seed):
    n_bound = boundary.sum()
    print(f'\n{"="*68}\n{title}\n{"="*68}')
    print(f'Amostras: {X.shape[0]} | Ruído: {NOISE_STD:.1e} | Seed: {seed} '
          f'| Boundary: {n_bound} ({100*n_bound/len(y):.1f}%)')
    print(f'{"Classe":20s}  {"N":>5s}  {"B médio":>8s}  {"f médio":>7s}  '
          f'{"SNR_V":>7s}  {"Boundary":>8s}  Status')
    print('-' * 70)
    for b_min, b_max, f_min, f_max, lid, name in CLASS_DEFS:
        mask  = (y == lid)
        B_m   = np.mean(B[mask])
        F_m   = np.mean(F[mask])
        n_b   = boundary[mask].sum()
        snr   = np.ptp(get_stokes(B_m, filling=F_m)['V']) / NOISE_STD
        status = ('ESPERADO' if lid == 0 else
                  'CRÍTICO'  if snr < 2  else
                  'DESAFIO'  if snr < 10 else 'NOMINAL')
        print(f'{name:20s}  {mask.sum():>5d}  {B_m:>8.1f}  {F_m:>7.3f}  '
              f'{snr:>7.1f}  {n_b:>8d}  {status}')
    orfaos = ((F > 0.20) & (F < 0.30) & (y == 0)).sum()
    print(f'\n  Amostras f∈(0.20,0.30) rotuladas Quieta : {orfaos}  (esperado 0)')


if __name__ == "__main__":
    print('✓ CLASS_DEFS — fronteiras contíguas sem gap.')
    print('Gerando Dataset Master...')

    (X_static,  y_static,  B_all, F_all,
     B_norm_all, F_norm_all, boundary_all) = generate_master_dataset(
        NOISE_STD, SEED_BASE, 'static')

    (X_doppler, y_doppler, _, _,
     B_norm_dop, F_norm_dop, boundary_dop) = generate_master_dataset(
        NOISE_STD, SEED_BASE + 1, 'doppler')

    (X_grad, y_grad, _, _,
     B_norm_grad, F_norm_grad, boundary_grad) = generate_master_dataset(
        NOISE_STD, SEED_BASE + 2, 'gradient')

    print_master_report('BLOCO A — Estático',
                        X_static,  y_static,  B_all, F_all, boundary_all, SEED_BASE)
    print_master_report('BLOCO B — Doppler',
                        X_doppler, y_doppler, B_all, F_all, boundary_dop, SEED_BASE+1)
    print_master_report('BLOCO C — Gradiente',
                        X_grad,    y_grad,    B_all, F_all, boundary_grad, SEED_BASE+2)

    assert B_norm_all.min() >= 0.0 and B_norm_all.max() <= 1.0
    assert F_norm_all.min() >= 0.0 and F_norm_all.max() <= 1.0
    for lid in [0, 1, 2]:
        assert (y_static == lid).sum() == N_SAMPLES

    print(f'\n✓ Célula 4 concluída.')
    print(f'  Novos arrays: B_norm_all, F_norm_all, boundary_all')