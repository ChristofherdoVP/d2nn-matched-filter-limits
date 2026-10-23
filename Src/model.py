"""
model.py — Configuração e arquitetura da D²NN (Difractive Deep Neural
Network) para classificação de perfis Stokes Ca II 854.2 nm.

Esta primeira seção (hiperparâmetros, tensores ópticos, regiões do
detector) corresponde à Célula 15 do notebook original. As classes do
modelo (CamadaFase, SpectralDetector, D2NN) são adicionadas na
próxima etapa (Célula 16).
"""

import numpy as np
import torch

from Src.config import DEVICE, DTYPE
from Src.physics import CALIB, LAM
from Src.optics import SYS4F, PUPIL, FX, FY, FX_NORM, FY_NORM

# =============================================================================
# CONSTANTES DO BLOCO 3 — Arquitetura e Treinamento
# Centralizado aqui — alterar apenas neste bloco.
# =============================================================================
B3 = {
    # --- Arquitetura D²NN ---
    'n_layers'    : 3,        # número de camadas de fase aprendíveis
    'det_regions' : 3,        # regiões do detector (uma por classe)

    # --- Treinamento ---
    'batch_size'  : 32,
    'lr'          : 1e-4,
    'epochs_cpu'  : 50,
    'epochs_gpu'  : 250,
    'val_frac'    : 0.15,
    'test_frac'   : 0.15,

    # --- Regularização ---
    'lr_patience' : 20,
    'lr_factor'   : 0.5,
    'weight_decay': 1e-4,

    # --- Nomes das classes (consistente com CLASS_DEFS de dataset.py) ---
    'class_names' : ['Região Quieta', 'Rede Magnética', 'Mancha Solar'],
}

# Número de épocas efetivo: ajusta automaticamente ao dispositivo
B3['epochs'] = B3['epochs_gpu'] if DEVICE.type == 'cuda' else B3['epochs_cpu']

# =============================================================================
# CONVERSÃO DAS CONSTANTES ÓPTICAS PARA TENSORES PYTORCH
# Feita UMA vez aqui — reutilizadas em todo o Bloco 3.
# =============================================================================
N_LAM_T = CALIB['n_lam']

PUPIL_T   = torch.tensor(PUPIL,     dtype=DTYPE, device=DEVICE)
FX_T      = torch.tensor(FX,        dtype=DTYPE, device=DEVICE)
FY_T      = torch.tensor(FY,        dtype=DTYPE, device=DEVICE)
FX_NORM_T = torch.tensor(FX_NORM,   dtype=DTYPE, device=DEVICE)
FY_NORM_T = torch.tensor(FY_NORM,   dtype=DTYPE, device=DEVICE)

# Inicialização da Camada 1: Vórtice m=1 (melhor candidato do ranking de máscaras)
# As camadas seguintes são inicializadas com fase zero (identidade)
PHI_VORTICE = torch.tensor(
    np.arctan2(FY, FX).astype(np.float32),
    dtype=DTYPE, device=DEVICE
)

assert PUPIL_T.shape == (SYS4F['det_pixels'], N_LAM_T), \
    f"ERRO: shape da pupila {PUPIL_T.shape} inconsistente."
assert FX_T.shape == PUPIL_T.shape, \
    f"ERRO: shape de FX_T {FX_T.shape} inconsistente com PUPIL_T."

# =============================================================================
# REGIÕES DO DETECTOR
# O plano de saída é dividido em 3 faixas no EIXO Y (paramétrico em B).
# Física: cada regime ocupa uma faixa de B no espectrograma de entrada,
# e a D²NN aprende a preservar essa separação no plano de saída.
# O eixo x (espectral) é integrado completamente — toda a largura passa.
# =============================================================================
DET_PX_OUT    = SYS4F['det_pixels'] * SYS4F['pad_factor']
REGION_SIZE_Y = DET_PX_OUT // B3['det_regions']

DETECTOR_REGIONS = {
    0: (slice(0,                  REGION_SIZE_Y),     slice(None)),  # Região Quieta
    1: (slice(REGION_SIZE_Y,      2 * REGION_SIZE_Y), slice(None)),  # Rede Magnética
    2: (slice(2 * REGION_SIZE_Y,  DET_PX_OUT),         slice(None)),  # Mancha Solar
}

import torch.nn as nn

# =============================================================================
# ARQUITETURA D²NN
# Corresponde à Célula 16 do notebook original.
#
# PROBLEMA CORRIGIDO NO ORIGINAL:
#   I3.mean(dim=2) colapsava os pontos espectrais num único valor médio
#   por linha. O sinal Zeeman (lóbulos ±ΔλB antissimétricos) integra a
#   zero — a informação que diferencia as classes era destruída antes
#   do classificador.
#
# SOLUÇÃO — SpectralDetector (Conv1d aprendível):
#   Banco de filtros espectrais aprendíveis que detecta padrões locais
#   (lóbulo azul, lóbulo vermelho, gradiente, assimetria) antes de
#   colapsar o eixo espectral com AdaptiveAvgPool1d.
#
# Referências:
#   Lin et al. (2018), Science 361, 1004-1008
#   Zhang et al. (2026), Nature Photonics — POMMM / tensor filter
# =============================================================================

SPEC_DET = {
    'n_filtros'      : 8,
    'kernel_spec'    : 31,
    'n_spectral_out' : 16,
}


class CamadaFase(nn.Module):
    """
    Camada de fase aprendível — propaga campo complexo sem detecção
    intermediária. Gradiente flui até delta_phi sem atenuação.
    Ref.: Lin et al. (2018).
    """

    def __init__(self, fase_init, pupila, pad=None):
        super().__init__()
        if pad is None:
            pad = SYS4F['pad_factor']
        self.pad       = pad
        self.pupila    = pupila
        self.delta_phi = nn.Parameter(torch.zeros_like(fase_init))
        self.register_buffer('fase_base', fase_init)

    def get_mascara(self):
        phi  = self.fase_base + self.delta_phi
        mask = torch.polar(torch.ones_like(phi), phi)
        return torch.complex(mask.real * self.pupila,
                             mask.imag * self.pupila)

    def forward(self, campo, aplicar_pad=False):
        E  = campo.to(torch.complex64) if not campo.is_complex() else campo
        ny, nx = E.shape[-2], E.shape[-1]

        F            = torch.fft.fft2(E, norm='ortho')
        mask_unshift = torch.fft.ifftshift(self.get_mascara(), dim=(-2, -1))
        F            = F * mask_unshift

        if aplicar_pad and self.pad > 1:
            new_ny, new_nx = ny * self.pad, nx * self.pad
            F_pad = torch.zeros((*F.shape[:-2], new_ny, new_nx),
                                dtype=torch.complex64, device=F.device)
            sy = (new_ny - ny) // 2
            sx = (new_nx - nx) // 2
            F_pad[..., sy:sy+ny, sx:sx+nx] = F
            F = F_pad

        return torch.fft.ifft2(F, norm='ortho')


class SpectralDetector(nn.Module):
    """
    Banco de filtros espectrais aprendíveis.

    Recebe I3 (batch, n_linhas, n_lambda) e produz features
    (batch, n_linhas * n_filtros * n_spectral_out).

    Cada linha do detector é processada por Conv1d com os mesmos pesos
    (compartilhamento espacial) — fisicamente correto porque o filtro
    óptico é espacialmente uniforme na pupila.
    """

    def __init__(self, n_filtros, kernel_spec, n_spectral_out):
        super().__init__()
        pad_conv = kernel_spec // 2
        self.filtros_spec = nn.Conv1d(
            in_channels  = 1,
            out_channels = n_filtros,
            kernel_size  = kernel_spec,
            padding      = pad_conv,
            bias         = True,
        )
        self.relu    = nn.ReLU()
        self.pool    = nn.AdaptiveAvgPool1d(n_spectral_out)
        self.flatten = nn.Flatten()

    def forward(self, I3):
        batch, n_lin, n_lam = I3.shape
        x = I3.reshape(batch * n_lin, 1, n_lam)
        x = self.filtros_spec(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.reshape(batch, n_lin, -1)
        return self.flatten(x)


class D2NN(nn.Module):
    """
    D²NN com propagação coerente e detector espectral aprendível.

    Fluxo:
      (batch, 64, 3000)
        → camada1,2     → campo complexo (batch, 64, 3000)
        → camada3       → campo complexo (batch, 128, 6000)  ← pad aqui
        → |E|²          → intensidade    (batch, 128, 6000)
        → SpectralDetector               (batch, 128*8*16 = 16384)
        → BN → Linear(16384,256) → ReLU → Dropout(0.3)
        → Linear(256,64)         → ReLU → Dropout(0.2)
        → Linear(64,3)           → logits
    """

    def __init__(self, pupila, phi_vortice, pad=None,
                 n_filtros=None, kernel_spec=None, n_spectral_out=None):
        super().__init__()
        if pad is None:
            pad = SYS4F['pad_factor']

        n_f = n_filtros      or SPEC_DET['n_filtros']
        k_s = kernel_spec    or SPEC_DET['kernel_spec']
        n_s = n_spectral_out or SPEC_DET['n_spectral_out']

        self.camada1 = CamadaFase(phi_vortice, pupila, pad)
        fase_zero    = torch.zeros_like(phi_vortice)
        self.camada2 = CamadaFase(fase_zero, pupila, pad)
        self.camada3 = CamadaFase(fase_zero, pupila, pad)

        self.detector_spec = SpectralDetector(n_f, k_s, n_s)

        dim_in = DET_PX_OUT * n_f * n_s
        self.classificador = nn.Sequential(
            nn.BatchNorm1d(dim_in),
            nn.Linear(dim_in, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        E1    = self.camada1(x,  aplicar_pad=False)
        E2    = self.camada2(E1, aplicar_pad=False)
        E3    = self.camada3(E2, aplicar_pad=True)
        I3    = E3.abs() ** 2
        feats = self.detector_spec(I3)
        return self.classificador(feats)

    def get_fases(self):
        return {
            'camada1': (self.camada1.fase_base
                        + self.camada1.delta_phi).detach().cpu().numpy(),
            'camada2': (self.camada2.fase_base
                        + self.camada2.delta_phi).detach().cpu().numpy(),
            'camada3': (self.camada3.fase_base
                        + self.camada3.delta_phi).detach().cpu().numpy(),
        }

    def get_kernels_espectrais(self):
        """Retorna kernels aprendidos para visualização e interpretação física."""
        with torch.no_grad():
            k = self.detector_spec.filtros_spec.weight.cpu().numpy()
        return k[:, 0, :]

# =============================================================================
# propagate_4f em PyTorch (diagnóstico/validação cruzada)
# Corresponde à Célula 18 do notebook original.
#
# Esta função NÃO é usada no forward da D²NN (que usa CamadaFase
# diretamente) — serve para validar a implementação diferenciável
# contra a versão NumPy já testada (optics.propagate_4f).
# =============================================================================

def propagate_4f_torch(field, mask=None, pad=None):
    """
    Propaga um campo 2D através de um sistema 4f — versão PyTorch.

    Parâmetros
    ----------
    field : Tensor real ou complexo (batch, ny, nx) ou (ny, nx)
    mask  : Tensor complexo (ny, nx) ou None — máscara no plano de Fourier.
            Se None, aplica apenas PUPIL_T.
    pad   : int ou None — fator de zero-padding. Se None, usa SYS4F['pad_factor'].

    Retorna
    -------
    Tensor complexo — campo no plano imagem. Use .abs()**2 para intensidade.
    """
    if pad is None:
        pad = SYS4F['pad_factor']

    squeeze = field.dim() == 2
    if squeeze:
        field = field.unsqueeze(0)

    if not field.is_complex():
        E_in = field.to(torch.complex64)
    else:
        E_in = field

    ny, nx = E_in.shape[-2], E_in.shape[-1]

    F = torch.fft.fftshift(
            torch.fft.fft2(E_in, norm='ortho'),
            dim=(-2, -1))

    if mask is None:
        F_filtered = F * PUPIL_T.to(torch.complex64)
    else:
        F_filtered = F * mask

    if pad > 1:
        new_ny, new_nx = ny * pad, nx * pad
        F_padded = torch.zeros(
            (*F_filtered.shape[:-2], new_ny, new_nx),
            dtype=torch.complex64, device=F_filtered.device)
        sy = (new_ny - ny) // 2
        sx = (new_nx - nx) // 2
        F_padded[..., sy:sy+ny, sx:sx+nx] = F_filtered
        F_filtered = F_padded

    E_out = torch.fft.ifft2(
                torch.fft.ifftshift(F_filtered, dim=(-2, -1)),
                norm='ortho')

    if squeeze:
        E_out = E_out.squeeze(0)

    return E_out


# =============================================================================
# D²NN com campo de referência coerente (detecção homódina)
# =============================================================================

class D2NNHomodina(D2NN):
    """
    D²NN com um campo de referência coerente somado à entrada.

    Motivação física: a detecção final mede |E|². Sem referência, um sinal
    fraco s aparece como |s|² (detecção quadrática, de energia), que perde
    para a detecção coerente quando o sinal está imerso em ruído. Com um
    campo de referência uniforme β, após o sistema 4f:

        |β·e^{iψ} + s|² = β² + 2β·Re(e^{-iψ}·s) + |s|²

    e o termo cruzado é LINEAR no sinal (detecção homódina). A referência
    uniforme ocupa só a frequência zero no plano de Fourier; a fase ψ é a
    fase aprendida da máscara nesse pixel (fase do "oscilador local").

    β é aprendível e começa em `beta_init` (padrão 0): com β = 0 o modelo
    é idêntico à D2NN base, então dá para partir de um checkpoint existente
    e deixar o treino decidir se a referência ajuda.
    """

    def __init__(self, *args, beta_init=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_ref = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x):
        return super().forward(x + self.beta_ref)


# =============================================================================
# D²NN puramente óptica (script 62)
# =============================================================================

class D2NNOptica(nn.Module):
    """
    Uma máscara de fase no plano de Fourier + energia numa janela + 2 limiares.

    X (64 × N_LAM) → CamadaFase (sem padding) → |E|² → roll (origem → centro) →
    E = soma numa janela (linhas × colunas) → z = (log E − μ)/s →
    logits_k = w_k·z + b_k.

    A "eletrônica" são 6 números (w, b): dois limiares ordinais sobre a energia de
    um único detector. Com as classes de mesma forma (achado 18), toda leitura
    óptica se reduz a um escalar; o ótimo é concentrar a projeção casada num ponto,
    o que o filtro casado de fase (POF, H = S*/|S|) faz levando o sinal à origem.
    Invariante à polaridade por construção (|E|² é par em V).

    diafragma_pm (opcional, script 67): diafragma de campo no plano de entrada — só passa
    |λ − λ_núcleo| ≤ diafragma_pm. Uma máscara de fase pura tem ganho 1 em todas as
    frequências e não atenua o ruído longe da linha (o filtro casado atenua); o diafragma
    faz isso opticamente. None = sem diafragma (checkpoints anteriores).
    """

    def __init__(self, fase_init, janela, treinar_fase=True, diafragma_pm=None):
        super().__init__()
        self.diafragma_pm = diafragma_pm
        if diafragma_pm is not None:
            m = np.abs((LAM - CALIB['lambda0_core']) * 1e3) <= float(diafragma_pm)
            self.register_buffer('diafragma', torch.tensor(m.astype(np.float32), device=PUPIL_T.device))
        else:
            self.diafragma = None
        self.mascara = CamadaFase(fase_init, PUPIL_T, pad=1)
        self.mascara.delta_phi.requires_grad_(treinar_fase)
        self.janela = tuple(janela)
        self.w = nn.Parameter(torch.tensor([-1.0, 0.0, 1.0]))
        self.b = nn.Parameter(torch.zeros(3))
        self.register_buffer('mu', torch.tensor(0.0))
        self.register_buffer('s', torch.tensor(1.0))

    def energia(self, x):
        if self.diafragma is not None:
            x = x * self.diafragma
        E = self.mascara(x, aplicar_pad=False)
        I = E.abs() ** 2
        ny, nx = I.shape[-2], I.shape[-1]
        I = torch.roll(I, shifts=(ny // 2, nx // 2), dims=(-2, -1))
        hy, hx = self.janela[0] // 2, self.janela[1] // 2
        return I[..., ny // 2 - hy: ny // 2 + hy + 1,
                 nx // 2 - hx: nx // 2 + hx + 1].sum(dim=(-2, -1))

    def forward(self, x):
        z = (torch.log(self.energia(x) + 1e-12) - self.mu) / self.s
        return self.w[None, :] * z[:, None] + self.b[None, :]


class CamadaComplexa(nn.Module):
    """
    Máscara de amplitude e fase no plano de Fourier (script 74): t = a·e^{iφ}, 0 < a < 1
    (elemento passivo — SLM de fase + máscara de absorção). a = sigmoid(ρ), φ = φ₀ + Δφ.
    Mesma convenção de CamadaFase (máscara em coordenadas centradas, × pupila, ifftshift).
    """

    def __init__(self, fase_init, amp_init, pupila):
        super().__init__()
        self.pupila = pupila
        self.register_buffer('fase_base', fase_init)
        self.delta_phi = nn.Parameter(torch.zeros_like(fase_init))
        a = torch.clamp(amp_init, 1e-3, 1 - 1e-3)
        self.rho = nn.Parameter(torch.log(a / (1 - a)))

    def amplitude(self):
        return torch.sigmoid(self.rho)

    def get_mascara(self):
        mask = torch.polar(self.amplitude(), self.fase_base + self.delta_phi)
        return torch.complex(mask.real * self.pupila, mask.imag * self.pupila)

    def forward(self, campo, aplicar_pad=False):
        E = campo.to(torch.complex64) if not campo.is_complex() else campo
        F = torch.fft.fft2(E, norm='ortho') * torch.fft.ifftshift(self.get_mascara(), dim=(-2, -1))
        return torch.fft.ifft2(F, norm='ortho')


class D2NNOpticaComplexa(D2NNOptica):
    """
    D2NNOptica com máscara de amplitude e fase (CamadaComplexa) no lugar da máscara só de fase.
    Script 73: no ViSP a máscara só de fase não passa de 1,56× o filtro casado; a máscara
    complexa H = S* reproduz o filtro (0,99–1,00×). Leitura e limiares iguais aos da D2NNOptica.
    """

    def __init__(self, fase_init, amp_init, janela, treinar_mascara=True, diafragma_pm=None):
        super().__init__(fase_init, janela, treinar_fase=False, diafragma_pm=diafragma_pm)
        self.mascara = CamadaComplexa(fase_init, amp_init, PUPIL_T)
        for prm in self.mascara.parameters():
            prm.requires_grad_(treinar_mascara)


if __name__ == "__main__":
    print(f'✓ PyTorch {torch.__version__}')
    print(f'  Dispositivo : {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'  GPU         : {torch.cuda.get_device_name(0)}')
        print(f'  VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

    print(f'\n{"="*60}')
    print('CONFIGURAÇÃO DO BLOCO 3 — D²NN')
    print(f'{"="*60}')
    print(f'  Arquitetura     : {B3["n_layers"]} camadas de fase aprendíveis')
    print(f'  Épocas          : {B3["epochs"]}  '
          f'({"GPU" if DEVICE.type == "cuda" else "CPU"} mode)')
    print(f'  Batch size      : {B3["batch_size"]}')
    print(f'  Learning rate   : {B3["lr"]}')
    print(f'\n  Regiões do detector (eixo y, {DET_PX_OUT} linhas após pad):')
    for cls, (slc_y, slc_x) in DETECTOR_REGIONS.items():
        print(f'    Classe {cls} ({B3["class_names"][cls]:15s}): '
              f'pixels {slc_y.start}–{slc_y.stop}')
    print(f'{"="*60}')

    # --- Instanciar e verificar ---
    modelo = D2NN(pupila=PUPIL_T, phi_vortice=PHI_VORTICE).to(DEVICE)

    n_fase  = sum(p.numel() for p in [*modelo.camada1.parameters(),
                                       *modelo.camada2.parameters(),
                                       *modelo.camada3.parameters()]
                  if p.requires_grad)
    n_det   = sum(p.numel() for p in modelo.detector_spec.parameters()
                  if p.requires_grad)
    n_cls   = sum(p.numel() for p in modelo.classificador.parameters()
                  if p.requires_grad)
    n_total = sum(p.numel() for p in modelo.parameters() if p.requires_grad)

    dim_in = DET_PX_OUT * SPEC_DET['n_filtros'] * SPEC_DET['n_spectral_out']

    print('\n✓ Célula 16 — D²NN instanciada.')
    print(f'\n{"="*62}')
    print('ARQUITETURA D²NN')
    print(f'{"="*62}')
    print(f'  Camadas de fase    : 3  ({n_fase:,} params)')
    print(f'  SpectralDetector   : Conv1d({SPEC_DET["n_filtros"]} filtros, '
          f'kernel={SPEC_DET["kernel_spec"]} pts ≈ '
          f'{SPEC_DET["kernel_spec"]/3:.0f} mÅ)  ({n_det} params)')
    print(f'  Pool espectral     : AdaptiveAvgPool1d({SPEC_DET["n_spectral_out"]} pts)')
    print(f'  Classificador      : {dim_in} → 256 → 64 → 3  ({n_cls:,} params)')
    print(f'  Total              : {n_total:,} params')
    print(f'  Dispositivo        : {DEVICE}')
    print(f'{"="*62}')

    with torch.no_grad():
        _x  = torch.rand(4, SYS4F['det_pixels'], CALIB['n_lam'], device=DEVICE)
        _lg = modelo(_x)
        assert _lg.shape == (4, 3), f"ERRO shape: {_lg.shape}"

    print(f'\n  Forward pass (batch=4):')
    print(f'    Entrada  : {tuple(_x.shape)}')
    print(f'    Logits   : {tuple(_lg.shape)}  ✓')
    print(f'    Escala   : {_lg.abs().mean().item():.4f}')

    k = modelo.get_kernels_espectrais()
    print(f'\n  Kernels espectrais : {k.shape}  '
          f'(mean={k.mean():.4f}, std={k.std():.4f})')
    print(f'\n✓ Pronto para o treino (Bloco 3).')