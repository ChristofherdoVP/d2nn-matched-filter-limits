"""
config.py — Configurações centrais do projeto PHOTON ROAD (D²NN Ca II 854.2 nm)

Substitui o setup que antes estava espalhado em várias células do notebook
Colab (Célula 0, e as redefinições de DRIVE_PATH/DEVICE ao longo do notebook).
Todo outro módulo do projeto deve importar as constantes daqui, em vez de
redefini-las.
"""

import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# 1. Caminhos do projeto (substitui o antigo PATH_ROOT + drive.mount)
# -----------------------------------------------------------------------
# BASE_DIR = pasta raiz do projeto (duas pastas acima deste arquivo: Src/config.py -> raiz)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR      = BASE_DIR / "Data"
RAW_SST_DIR   = DATA_DIR / "RAW_SST"     # dados reais (.fits) — hoje no SSD, HD externo no futuro
REFERENCE_DIR = DATA_DIR / "Reference"
SYNTHETIC_DIR = DATA_DIR / "Synthetic"   # perfis Stokes sintéticos gerados localmente
PROCESSED_DIR = DATA_DIR / "Processed"   # .npy intermediários (X_4f_norm, y_static, etc.)
H5_DIR        = DATA_DIR / "H5"          # arquivo consolidado photon_road.h5

CHECKPOINTS_DIR = BASE_DIR / "CheckPoints"
OUTPUTS_DIR     = BASE_DIR / "Outputs"
FIGURES_DIR     = OUTPUTS_DIR / "Figures"
REPORTS_DIR     = OUTPUTS_DIR / "Reports"
# Garante que todas as pastas existem (não sobrescreve as que você já criou)
for _dir in (RAW_SST_DIR, REFERENCE_DIR, SYNTHETIC_DIR, PROCESSED_DIR, H5_DIR,
             CHECKPOINTS_DIR, OUTPUTS_DIR, FIGURES_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------
# 2. Reprodutibilidade
# -----------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------------------------------------------------------
# 3. Hardware
# -----------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32   # float32 é suficiente e mais rápido que float64
# -----------------------------------------------------------------------
# 4. Avisos
# -----------------------------------------------------------------------
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------------------------------------------------
# 5. Estilo visual (tema escuro GitHub)
# -----------------------------------------------------------------------
plt.rcParams.update({
    'figure.dpi'       : 100,
    'font.family'      : 'serif',
    'axes.facecolor'   : '#0d1117',
    'figure.facecolor' : '#0d1117',
    'axes.edgecolor'   : '#30363d',
    'axes.labelcolor'  : '#e6edf3',
    'xtick.color'      : '#8b949e',
    'ytick.color'      : '#8b949e',
    'grid.color'       : '#21262d',
    'grid.linewidth'   : 0.6,
    'text.color'       : '#e6edf3',
    'axes.titlecolor'  : '#e6edf3',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

# -----------------------------------------------------------------------
# 6. Paleta de cores centralizada
# -----------------------------------------------------------------------
COLORS = {
    'I'       : '#79c0ff',
    'V'       : '#ff7b72',
    'Q'       : '#56d364',
    'U'       : '#ffa657',
    'atlas'   : '#bc8cff',
    'ref'     : '#8b949e',
    'zero'    : '#30363d',
    'inversao': '#d29922',
    'regimes' : {'quieta': '#8b949e', 'rede': '#79c0ff', 'mancha': '#ffa657'},
}

# -----------------------------------------------------------------------
# 7. Diagnóstico — só roda se você executar este arquivo diretamente
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"✅ Executando em: {DEVICE}")
    print(f"📂 BASE_DIR       : {BASE_DIR}")
    print(f"📂 RAW_SST_DIR    : {RAW_SST_DIR}")
    print("🚀 PHOTON ROAD base pronta para processamento.")