"""
optics.py — Funções geométricas e de propagação para o sistema óptico 4f
da D²NN (grades 2D, máscaras de fase, matrizes DFT explícitas).

Formalismo: cada CamadaFase implementa FT → Hadamard(H_k) → FT, análogo
ao pipeline POMMM (Zhang et al., "Direct tensor processing with coherent
light", Nature Photonics 20, 102-108, 2026, DOI: 10.1038/s41566-025-01799-7).

Corresponde às funções auxiliares da Célula 8B do notebook original.
"""

import numpy as np
import torch

from Src.config import DEVICE


def grade_2d(N, device=DEVICE):
    """Retorna (X, Y, R, PHI) normalizados em [-1,1]."""
    x = torch.linspace(-1, 1, N, device=device)
    X, Y = torch.meshgrid(x, x, indexing='xy')
    R    = torch.sqrt(X**2 + Y**2)
    PHI  = torch.atan2(Y, X)
    return X, Y, R, PHI


def fazer_mascaras(N, device=DEVICE):
    """Pupila circular, vórtice m=1, H2=H3=pupila."""
    _, _, R, PHI = grade_2d(N, device)
    pupila = (R <= 0.95).float()
    H1 = torch.exp(1j * PHI).to(torch.complex64) * pupila
    H2 = pupila.to(torch.complex64)
    H3 = pupila.to(torch.complex64)
    return pupila, H1, H2, H3


def propagar_4f_step(E, H):
    """Uma passagem: fftshift(fft2) → máscara → ifftshift(fft2)."""
    F        = torch.fft.fftshift(torch.fft.fft2(E))
    F_masked = F * H
    E_out    = torch.fft.fftshift(torch.fft.fft2(
                   torch.fft.ifftshift(F_masked)))
    return E_out


def dft_matrix_1d_shifted(N, device=DEVICE):
    """
    Matriz DFT 1D com convenção fftshift:
    D[k,n] = exp(-2πi·k·n/N) × (-1)^(k+n)
    O fator (-1)^(k+n) implementa o fftshift/ifftshift.
    """
    k   = torch.arange(N, device=device, dtype=torch.float32)
    n   = torch.arange(N, device=device, dtype=torch.float32)
    D   = torch.exp(-2j * np.pi * torch.outer(k, n) / N).to(torch.complex64)
    sk  = ((-1.0) ** k).to(torch.complex64)
    sn  = ((-1.0) ** n).to(torch.complex64)
    return D * sk.unsqueeze(1) * sn.unsqueeze(0)


def dft_matrix_2d_shifted(N, device=DEVICE):
    """
    Matriz DFT 2D com fftshift: kronecker(D1d, D1d)
    Tamanho: (N^2, N^2) — usar apenas para N pequeno.
    """
    D1 = dft_matrix_1d_shifted(N, device)
    return torch.kron(D1, D1)


def mascara_diag(H, device=DEVICE):
    """Transforma máscara 2D (N,N) em matriz diagonal (N^2, N^2)."""
    return torch.diag(H.reshape(-1).to(torch.complex64))

# -----------------------------------------------------------------------
# Sistema Óptico 4f — Parâmetros e Grade de Fourier
# Corresponde à Célula 9 do notebook original.
#
# A pupila é elíptica assimétrica: estreita no eixo espectral (x), para
# capturar a forma de Stokes V; aberta no eixo paramétrico (y), para
# preservar a variação de amplitude de V com o campo B.
# -----------------------------------------------------------------------

from Src.physics import CALIB, DLAM, FWHM_SOLAR

N_LAM = CALIB['n_lam']   # 3000 pontos espectrais

SYS4F = {
    # --- Detector ---
    'det_pixels' : 64,
    'B_max'      : 2500.0,

    # --- Óptica: pupila elíptica assimétrica ---
    'NA_x'       : 0.02,     # corte espectral [fração do Nyquist de x]
    'NA_y'       : 1.00,     # sem corte paramétrico [fração do Nyquist de y]

    'pad_factor' : 2,

    # --- Modulação polarimétrica ---
    'qwp_angle'  : 45.0,
    'pol_angle'  : 0.0,      # modo LCP: I_det = 0.5(I + V)

    # --- Regimes de referência para diagnóstico e ranking de máscaras ---
    'regimes' : {
        'quieta': {'B': 50,   'f': 0.025, 'label': 'Região Quieta'},
        'rede'  : {'B': 300,  'f': 0.125, 'label': 'Rede Magnética'},
        'mancha': {'B': 1500, 'f': 0.650, 'label': 'Mancha Solar'  },
    },
}

det_px = SYS4F['det_pixels']
NA_x   = SYS4F['NA_x']
NA_y   = SYS4F['NA_y']

# Frequências espectrais [ciclos/nm] — eixo x
FX_1D = np.fft.fftshift(np.fft.fftfreq(N_LAM, d=DLAM))
FX_NY = FX_1D.max()

# Frequências paramétricas [adimensional] — eixo y
FY_1D = np.fft.fftshift(np.fft.fftfreq(det_px, d=1.0))
FY_NY = FY_1D.max()

# Grades 2D — shape (det_px, N_LAM)
FX, FY = np.meshgrid(FX_1D, FY_1D)

# Frequências normalizadas [0, 1] em cada eixo
FX_NORM = np.abs(FX) / FX_NY
FY_NORM = np.abs(FY) / FY_NY

# Pupila elíptica assimétrica
PUPIL = ((FX_NORM < NA_x) & (FY_NORM < NA_y)).astype(np.float64)

# --- Diagnóstico da grade (calculado sempre, usado nos asserts) ---
f_V_max  = 1.0 / (FWHM_SOLAR / 1000.0 / 2.0)   # [ciclos/nm]
f_V_norm = f_V_max / FX_NY
f_cut_x  = NA_x * FX_NY

assert f_cut_x > f_V_max, \
    (f"ERRO: NA_x={NA_x} corta em {f_cut_x:.1f} c/nm, "
     f"abaixo de f_V_max={f_V_max:.1f} c/nm. Aumente NA_x.")
    
if __name__ == "__main__":
    print('✓ Célula 9 — Sistema 4f inicializado.')
    print(f'\n{"="*65}')
    print('PARÂMETROS DO SISTEMA 4f')
    print(f'{"="*65}')
    print(f'  Grade entrada       : {det_px} × {N_LAM} pixels')
    print(f'  Passo espectral     : {DLAM*1000:.4f} mÅ/pixel')
    print(f'  Nyquist espectral   : {FX_NY:.1f} ciclos/nm')
    print(f'  Zero-padding        : {SYS4F["pad_factor"]}x (interpolação digital)')
    print(f'\n  Pupila elíptica assimétrica:')
    print(f'    NA_x = {NA_x}  → corte em {f_cut_x:.1f} ciclos/nm  '
          f'({NA_x*100:.0f}% do Nyquist espectral)')
    print(f'    NA_y = {NA_y}  → sem corte paramétrico  '
          f'(variação de B preservada)')
    print(f'\n  Energia de V em f_x < {f_V_max:.1f} ciclos/nm  '
          f'({f_V_norm*100:.2f}% do Nyquist)')
    print(f'  Margem pupila/sinal : {f_cut_x/f_V_max:.1f}x  ✓')
    print(f'  Fração de pixels passantes (pupila): {PUPIL.mean()*100:.2f}%')
    print(f'\n  Assinaturas discriminantes capturadas simultaneamente:')
    print(f'    (1) Forma espectral de V  → eixo x, f_x < {f_cut_x:.1f} c/nm')
    print(f'    (2) Amplitude de V com B  → eixo y, variação paramétrica completa')
    print(f'\n  NOTA: splitting Zeeman acima do Nyquist para B < 1500G.')
    print(f'  O sistema discrimina por ENERGIA e ASSIMETRIA de V,')
    print(f'  não pelo splitting direto.')
    print(f'{"="*65}')
    
# -----------------------------------------------------------------------
# Espectrograma de Entrada do Sistema 4f
# Corresponde à Célula 10 do notebook original.
# -----------------------------------------------------------------------

from Src.physics import get_stokes, get_modulated_intensity, V_CORE_KMS


def build_stokes_spectrogram(theta_deg=0.0, chi_deg=0.0, filling=1.0,
                              B_map=None, qwp_angle_deg=None, pol_angle_deg=None):
    """
    Constrói o espectrograma 2D (det_pixels × N_LAM) varrendo o campo B
    linha a linha, para um dado ângulo/filling fixo.

    Retorna dict com 'img_I', 'img_V', 'img_Q', 'img_U', 'img_mod', 'B_map'.
    """
    if B_map is None:
        B_map = np.linspace(0, SYS4F['B_max'], SYS4F['det_pixels'])
    if qwp_angle_deg is None:
        qwp_angle_deg = SYS4F['qwp_angle']
    if pol_angle_deg is None:
        pol_angle_deg = SYS4F['pol_angle']

    det_px_local = len(B_map)
    img_I   = np.zeros((det_px_local, N_LAM), dtype=np.float64)
    img_V   = np.zeros((det_px_local, N_LAM), dtype=np.float64)
    img_Q   = np.zeros((det_px_local, N_LAM), dtype=np.float64)
    img_U   = np.zeros((det_px_local, N_LAM), dtype=np.float64)
    img_mod = np.zeros((det_px_local, N_LAM), dtype=np.float64)

    for i, B in enumerate(B_map):
        st = get_stokes(B, theta_deg=theta_deg, chi_deg=chi_deg,
                        filling=filling, v_km_s=V_CORE_KMS)
        I_det = get_modulated_intensity(st, qwp_angle_deg=qwp_angle_deg,
                                        pol_angle_deg=pol_angle_deg)
        img_I[i, :]   = st['I']
        img_V[i, :]   = st['V']
        img_Q[i, :]   = st['Q']
        img_U[i, :]   = st['U']
        img_mod[i, :] = I_det

    return {
        'img_I': img_I, 'img_V': img_V, 'img_Q': img_Q, 'img_U': img_U,
        'img_mod': img_mod, 'B_map': B_map,
    }


def subtract_dc(img_mod, method='first_row'):
    """
    Remove o componente DC dominante do espectrograma modulado, deixando
    visível a variação discriminante (sinal magnético).

    method='first_row' : subtrai a primeira linha (B=0), como no notebook
    method='mean'       : subtrai a média de todas as linhas (alternativa
                          comentada no notebook original)
    """
    if method == 'first_row':
        dc_ref = img_mod[0:1, :]
    elif method == 'mean':
        dc_ref = img_mod.mean(axis=0, keepdims=True)
    else:
        raise ValueError(f"method deve ser 'first_row' ou 'mean', recebido: {method}")
    return img_mod - dc_ref

# -----------------------------------------------------------------------
# Motor 4f (NumPy): propagate_4f
# Corresponde à Célula 11 do notebook original.
#
# Arquitetura: Lente L1 → [plano objeto] → FFT → [plano de Fourier: máscara]
# → IFFT → [plano imagem: field_out]
#
# Zero-padding aplicado no domínio de Fourier antes da IFFT (interpola a
# saída — mais pixels, mesma resolução física; não aumenta resolução
# angular do sistema).
# -----------------------------------------------------------------------

def propagate_4f(field, mask=None, pad=None, jitter_px=0):
    """
    Propaga um campo 2D através de um sistema 4f via FFT.

    Parâmetros
    ----------
    field : ndarray (ny, nx) — campo de entrada (intensidade modulada)
    mask  : ndarray (ny, nx) ou None — máscara no plano de Fourier.
            Se None, aplica apenas a pupila padrão (PUPIL).
            A máscara deve INCLUIR a pupila se desejado — não é aplicada
            automaticamente além da máscara.
    pad   : int ou None — fator de zero-padding. Se None, usa SYS4F['pad_factor'].
    jitter_px : int — erro de alinhamento mecânico simulado (deslocamento
                aleatório do campo antes da propagação)

    Retorna
    -------
    ndarray complexo (ny*pad, nx*pad) — campo no plano imagem.
    Use np.abs(out)**2 para intensidade.

    Física
    ------
    A normalização 1/sqrt(N) de cada FFT garante conservação de energia
    (norma de Parseval) para a máscara identidade.
    """
    if pad is None:
        pad = SYS4F['pad_factor']

    ny, nx = field.shape

    if jitter_px > 0:
        shift_y = np.random.randint(-jitter_px, jitter_px + 1)
        shift_x = np.random.randint(-jitter_px, jitter_px + 1)
        field = np.roll(field, shift=(shift_y, shift_x), axis=(0, 1))

    F = np.fft.fftshift(np.fft.fft2(field, norm='ortho'))

    F_filtered = F * (mask if mask is not None else PUPIL)

    if pad > 1:
        new_ny, new_nx = ny * pad, nx * pad
        F_padded = np.zeros((new_ny, new_nx), dtype=complex)
        sy, sx = (new_ny - ny) // 2, (new_nx - nx) // 2
        F_padded[sy:sy+ny, sx:sx+nx] = F_filtered
        F_filtered = F_padded

    return np.fft.ifft2(np.fft.ifftshift(F_filtered), norm='ortho')

# -----------------------------------------------------------------------
# Diagnóstico do Plano de Fourier por Regime Magnético
# Corresponde à Célula 12 do notebook original.
#
# Analisa onde a energia do sinal se concentra no espaço de Fourier,
# separando o espectrograma por regime (Quieta/Rede/Mancha). Os
# resultados (F_regimes, profiles_x) são reutilizados na Célula 13.
# -----------------------------------------------------------------------

REGIME_B_LIMITS = {
    'quieta': (0,    100),
    'rede'  : (100,  500),
    'mancha': (500, 2500),
}


def analyze_fourier_by_regime(field_in, B_map, regime_limits=None):
    """
    Separa o espectrograma de entrada por regime magnético (faixas de B),
    calcula o plano de Fourier de cada sub-espectrograma (com pupila
    aplicada) e mede a energia dentro/fora da banda espectral de V.

    Retorna: F_regimes, profiles_x, E_stats, regime_row_masks
    """
    if regime_limits is None:
        regime_limits = REGIME_B_LIMITS

    band_mask_x = np.abs(FX_1D) < f_V_max

    regime_row_masks = {}
    for regime, (b_lo, b_hi) in regime_limits.items():
        regime_row_masks[regime] = (B_map >= b_lo) & (B_map < b_hi)

    F_regimes  = {}
    profiles_x = {}
    E_stats    = {}

    for regime, rows in regime_row_masks.items():
        field_r = field_in[rows, :]

        F_r    = np.fft.fftshift(np.fft.fft2(field_r, norm='ortho'))
        F_filt = F_r * PUPIL[rows, :]
        F_pow  = np.abs(F_filt)**2

        F_regimes[regime] = F_r

        prof = F_pow.sum(axis=0)
        profiles_x[regime] = prof / (prof.max() + 1e-30)

        E_band  = F_pow[:, band_mask_x].sum()
        E_total = F_pow.sum()
        E_stats[regime] = {
            'E_band' : float(E_band),
            'E_total': float(E_total),
            'ratio'  : float(E_band / (E_total + 1e-30)),
            'n_rows' : int(rows.sum()),
            'B_range': regime_limits[regime],
        }

    return F_regimes, profiles_x, E_stats, regime_row_masks


# -----------------------------------------------------------------------
# Ranking de Máscaras: Banco de Candidatas e Avaliação Realista
# Corresponde à Célula 13 do notebook original.
#
# Testa candidatas a máscara de fase (vórtices, filtros casados, etc.)
# simulando efeitos de hardware real: quantização de fase do SLM
# (bits) e jitter mecânico de alinhamento (jitter_px).
# -----------------------------------------------------------------------

def compute_band_px(pad=None):
    """Largura em pixels (no detector com zero-padding) da banda espectral de V."""
    if pad is None:
        pad = SYS4F['pad_factor']
    nx_out  = N_LAM * pad
    fx_out  = np.fft.fftshift(np.fft.fftfreq(nx_out, d=DLAM))
    return int(np.sum(np.abs(fx_out) < f_V_max))


def build_mask_bank(field_in, regime_row_masks):
    """
    Constrói o banco de máscaras candidatas para a CamadaFase:
      - Vórtices topológicos (m=1, m=2)
      - Fase baseada em dI/dλ e no espectro de Stokes V
      - Filtro casado (Mancha - Quieta) e passa-banda puro
      - Pupila pura como referência

    Requer field_in (espectrograma pós-DC) e regime_row_masks (de
    analyze_fourier_by_regime) para as máscaras de engenharia.
    """
    VORTEX_1 = np.exp(1j * 1 * np.arctan2(FY, FX)) * PUPIL
    VORTEX_2 = np.exp(1j * 2 * np.arctan2(FY, FX)) * PUPIL

    _dI_dl_1d = np.gradient(get_stokes(0, v_km_s=V_CORE_KMS)['I'], DLAM)
    _dI_norm  = _dI_dl_1d / (np.max(np.abs(_dI_dl_1d)) + 1e-12)
    PHASE_DI  = np.exp(1j * np.pi * np.tile(_dI_norm, (SYS4F['det_pixels'], 1))) * PUPIL

    _st_v_ref = get_stokes(500, filling=1.0, v_km_s=V_CORE_KMS)['V']
    _F_V_ref  = np.fft.fftshift(np.fft.fft2(
        np.tile(_st_v_ref, (SYS4F['det_pixels'], 1)), norm='ortho'))
    PHASE_V   = np.exp(1j * np.angle(_F_V_ref)) * PUPIL

    _F_Q = np.fft.fftshift(np.fft.fft2(
        field_in[regime_row_masks['quieta'], :], norm='ortho')).mean(axis=0)
    _F_M = np.fft.fftshift(np.fft.fft2(
        field_in[regime_row_masks['mancha'], :], norm='ortho')).mean(axis=0)
    DIFF_FILTER = np.tile((_F_M - _F_Q), (SYS4F['det_pixels'], 1)) * PUPIL

    return {
        'Vórtice m=1'         : VORTEX_1,
        'Pupila (Referência)' : PUPIL.astype(complex),
        'Fase dI/dλ'          : PHASE_DI,
        'Fase espectro de V'  : PHASE_V,
        'Vórtice m=2'         : VORTEX_2,
        'Matched Filter (M-Q)': DIFF_FILTER,
        'Passa-banda puro'    : (FX_NORM < 0.02).astype(complex) * PUPIL,
    }


def score_discriminacao(field_in, regime_row_masks, mask, band_px,
                         pad=None, bits=8, jitter=1):
    """
    Avalia a separabilidade de uma máscara considerando hardware (bits
    de quantização de fase do SLM) e montagem (jitter_px de vibração).

    Retorna (results, sep): results = energia integrada por regime na
    janela central do detector; sep = separação Mancha - Quieta.
    """
    if pad is None:
        pad = SYS4F['pad_factor']

    if np.iscomplexobj(mask) and bits is not None:
        phase = np.angle(mask)
        levels = 2**bits
        phase_q = np.round(((phase + np.pi) / (2 * np.pi)) * (levels - 1)) / (levels - 1)
        phase_q = phase_q * 2 * np.pi - np.pi
        mask = np.exp(1j * phase_q) * np.abs(mask)

    results = {}
    for regime, rows in regime_row_masks.items():
        out = propagate_4f(field_in[rows, :], mask=mask[rows, :], pad=pad, jitter_px=jitter)
        out_int = np.abs(out)**2

        cx = out_int.shape[1] // 2
        hw = band_px // 2
        results[regime] = float(out_int[:, cx - hw : cx + hw].sum())

    sep = results['mancha'] - results['quieta']
    return results, sep

# -----------------------------------------------------------------------
# Interface Dataset → 4f: prepare_optical_input
# Corresponde à Célula 14 do notebook original.
#
# Converte perfis Stokes V (do dataset) em espectrogramas 2D normalizados,
# prontos para entrada da D²NN. Normalização Min-Max por amostra preserva
# a morfologia espectral de I+V, usada pela rede para discriminar classes.
# -----------------------------------------------------------------------

def prepare_optical_input(X_stokes_v, B_arr, F_arr, n_rows=None, verbose=True):
    """
    Converte perfis Stokes V com ruído em espectrogramas modulados 2D.

    Pipeline
    --------
    1. Recalcula I via motor canônico (sem ruído) para a amostra (B, f).
    2. Monta I_det = 0.5 * (I_clean + V_noisy)  — modulação LCP real.
    3. Normalização Min-Max por amostra → [0, 1].
    4. Broadcast para espectrograma 2D (n_rows × N_LAM).

    Parâmetros
    ----------
    X_stokes_v : ndarray (N, N_LAM) — perfis Stokes V com ruído (do dataset)
    B_arr      : ndarray (N,) — campos magnéticos [G]
    F_arr      : ndarray (N,) — filling factors
    n_rows     : int ou None — linhas do espectrograma (padrão: det_pixels)
    verbose    : bool — imprime progresso a cada 200 amostras

    Retorna
    -------
    X_4f    : ndarray (N, n_rows, N_LAM) float32 — espectrogramas em [0, 1]
    X_scale : float — valor máximo antes da normalização (para referência)
    """
    if n_rows is None:
        n_rows = SYS4F['det_pixels']

    N    = X_stokes_v.shape[0]
    X_4f = np.zeros((N, n_rows, N_LAM), dtype=np.float32)

    for i in range(N):
        st_i    = get_stokes(float(B_arr[i]), filling=float(F_arr[i]),
                             v_km_s=V_CORE_KMS)
        V_noisy = X_stokes_v[i].astype(np.float64)

        I_det = 0.5 * (st_i['I'] + V_noisy)

        i_min, i_max = I_det.min(), I_det.max()
        I_det = (I_det - i_min) / (i_max - i_min + 1e-8)

        X_4f[i] = np.tile(I_det, (n_rows, 1)).astype(np.float32)

        if verbose and (i + 1) % 200 == 0:
            print(f'  {i+1}/{N} amostras processadas...')

    return X_4f, float(X_4f.max())


# -----------------------------------------------------------------------
# Montagem da entrada da D²NN
#
# Duas receitas:
#   build_sweep_spectrograms      — ORIGINAL do Colab (depende de B e f).
#                                    Só para reproduzir o checkpoint antigo.
#   build_observable_spectrograms — OBSERVÁVEL (só perfis I, V). Usar no
#                                    treino e em dados reais.
# Ver docs/RECONSTRUCAO_NORMALIZACAO.md
# -----------------------------------------------------------------------

def build_sweep_spectrograms(B_arr, F_arr, n_rows=None, v_km_s=None, verbose=True):
    """
    Receita ORIGINAL do Colab (Células 25, 20B e 44B), reproduzida exatamente.

    ATENÇÃO: depende do B e do f VERDADEIROS de cada amostra. Cada linha é
    sintetizada com get_stokes(B_k, f), com B_k varrendo B ± 20% (piso de
    ±10 G). Como as classes são definidas por faixas de B e f, a entrada já
    contém a resposta — e não pode ser construída para dados observados.
    Mantida SOMENTE para reproduzir o checkpoint antigo (script 44).

    Retorna X_dc float32 (N, n_rows, N_LAM), sem escala global.
    """
    if n_rows is None:
        n_rows = SYS4F['det_pixels']
    if v_km_s is None:
        v_km_s = V_CORE_KMS

    N   = len(B_arr)
    X   = np.empty((N, n_rows, N_LAM), dtype=np.float32)
    raw = np.empty((n_rows, N_LAM), dtype=np.float32)

    for i in range(N):
        B, f = float(B_arr[i]), float(F_arr[i])
        dB   = max(B * 0.20, 10.0)
        B_ln = np.linspace(max(0.0, B - dB), B + dB, n_rows)
        for k, Bk in enumerate(B_ln):
            st = get_stokes(Bk, filling=f, v_km_s=v_km_s)
            raw[k] = 0.5 * (st['I'] + st['V'])
        X[i] = raw - raw.mean(axis=0, keepdims=True)
        if verbose and (i + 1) % 250 == 0:
            print(f'  {i+1}/{N} amostras processadas...')

    return X


# Fator de modulação ao longo das linhas: equivale à varredura B·(0.8 → 1.2)
# da receita original, no regime de campo fraco.
FATOR_RAMPA = (0.8, 1.2)


def build_observable_spectrograms(I_arr, V_arr, n_rows=None, fator_rampa=FATOR_RAMPA):
    """
    Entrada OBSERVÁVEL: monta o espectrograma apenas a partir dos perfis
    (I, V) — sem conhecer B ou f. Aplicável a dados reais.

    Linha k:  I_det,k = 0.5 * (I + a_k · V),  a_k de fator_rampa[0] a fator_rampa[1]
    seguida da subtração da média sobre as linhas. O resultado é
    0.5 · V · (a_k − ⟨a⟩): o termo de I se cancela exatamente, e a entrada
    carrega a informação polarimétrica (Stokes V).

    Em relação à receita original: idêntica para B ≥ 50 G (|r| ≈ 0,9998 na
    Rede, razão 1,000), mas sem a amplificação artificial dos campos fracos
    (B < 50 G) que a receita original introduzia.

    I_arr, V_arr : (N, N_LAM), na grade espectral LAM
    Retorna X_dc float32 (N, n_rows, N_LAM), sem escala global.
    """
    if n_rows is None:
        n_rows = SYS4F['det_pixels']

    a     = np.linspace(fator_rampa[0], fator_rampa[1], n_rows)
    I_arr = np.asarray(I_arr, dtype=np.float64)
    V_arr = np.asarray(V_arr, dtype=np.float64)

    N = I_arr.shape[0]
    X = np.empty((N, n_rows, I_arr.shape[1]), dtype=np.float32)
    for i in range(N):
        raw  = 0.5 * (I_arr[i][None, :] + a[:, None] * V_arr[i][None, :])
        X[i] = raw - raw.mean(axis=0, keepdims=True)

    return X
