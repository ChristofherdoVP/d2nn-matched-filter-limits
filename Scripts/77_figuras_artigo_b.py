"""
77_figuras_artigo_b.py — figuras do artigo B (Applied Optics), tema claro.

Lê os relatórios JSON dos scripts 61, 73, 75, 76 (Outputs/Reports) e os checkpoints
modelo_d2nn_optico_v2.pt (67) e modelo_d2nn_complexo_v2.pt (74) (CheckPoints).
Não treina nada e não gera estatísticas novas: só desenha.

Figuras (Outputs/Figures/artigo_b/, PNG 600 dpi + PDF):
  fig1_esquema       codificação X(y,λ) = c_y V(λ) e processador 4f (esquema)
  fig2_ablacao       ablação das D²NN híbridas 47, 57, 59 (script 61)
  fig3_pof           entrada X(y,λ), fase da máscara 67, intensidade de saída e janela 3 × 5
  fig4_razoes        a50 / ótimo contra k, IBIS, CRISP, ViSP (scripts 75 e 76)
  fig5_visp          ViSP: a50 / filtro casado contra k, máscaras fixas e treinada (script 73)
  fig6_complexa      amplitude da máscara 74 treinada e da projetada branqueada (CRISP);
                     CRISP: a50 / ótimo contra k (scripts 75 e 76)
  figS1_doppler      Suplemento 1: a50(Δv)/a50(0), um detector, fileira, filtro e banco (script 68)

Larguras: coluna simples 3,33 in, dupla 6,93 in; fonte 8 pt (Liberation Sans ≈ Arial).
Tempo: < 1 min (CPU).
"""
import sys
import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from Src.config import DEVICE, DTYPE
from Src.optics import FX, FY, PUPIL
from Src.physics import LAM, CALIB
from Src.evaluation import carregar_modelo, MontadorEntrada
from Src.train_utils import X_SCALE_TREINO

# ─────────────────────────── estilo (claro; config.py define tema escuro) ───────────────────────────
plt.rcdefaults()
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'lines.linewidth': 1.2, 'lines.markersize': 4,
    'legend.frameon': False, 'savefig.dpi': 600, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans',
    'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold',
})
W1, W2 = 3.33, 6.93
# Okabe–Ito (validado: scripts/validate_palette.js; pares próximos levam marcador distinto)
AZUL, VERM, VERDE, ROXO, PRETO, CINZA = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#000000', '#7A7A7A'

REP = ROOT / 'Outputs' / 'Reports'
CKPT = ROOT / 'CheckPoints'
OUT = ROOT / 'Outputs' / 'Figures' / 'artigo_b'
OUT.mkdir(parents=True, exist_ok=True)
INST = ['IBIS/DST', 'CRISP/SST', 'ViSP/DKIST']
INST_CURTO = ['IBIS', 'CRISP', 'ViSP']
K11, K44 = 1 / np.sqrt(11), 1 / np.sqrt(44)


def ler(nome):
    return json.load(open(REP / f'{nome}.json', encoding='utf-8'))['data']


def salvar(fig, nome):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{nome}.{ext}', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'  {nome}.png / .pdf')


def rotulo(ax, letra):
    ax.text(-0.02, 1.02, f'({letra})', transform=ax.transAxes, ha='right', va='bottom', fontweight='bold')


def eixo_k(ax):
    ax.set_xscale('log')
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2])
    ax.set_xticklabels(['0.1', '0.2', '0.5', '1', '2'])
    ax.minorticks_off()
    for k in (K44, K11):
        ax.axvline(k, color=CINZA, lw=0.5, ls=':', zorder=0)
    ax.axhline(1.0, color=CINZA, lw=0.6, zorder=0)
    ax.set_xlabel('Noise scale $k$')


def serie(lista):
    return np.array([e['k'] for e in lista]), np.array([e['bf50'] for e in lista])


# ═══════════════════════════════════ Fig. 2 ═══════════════════════════════════
def fig2():
    d = ler('61_ablacao_optica')
    modelos = [('47', 'Hybrid 1 (legacy noise)', AZUL, 'o'),
               ('57', 'Hybrid 2', VERM, 's'),
               ('59', 'Hybrid 3 (reference field)', VERDE, '^')]
    var = [('V1 vórtice estático', 'V1 no correction'), ('V2 só pupila', 'V2 pupil only'),
           ('V3 fase aleatória', 'V3 random phase'), ('V4 Δφ × 10', r'V4 correction $\times$10')]
    fig, axs = plt.subplots(1, 2, figsize=(W1, 1.55), gridspec_kw={'width_ratios': [1, 1.25], 'wspace': 0.08})
    for j, (chave, nome, cor, mk) in enumerate(modelos):
        ab = d['modelos'][chave]['ablacao']
        n = int(np.sum(ab['V0 treinado']['matriz']))
        a0 = ab['V0 treinado']['acuracia']
        for i, (v, _) in enumerate(var):
            mc = ab[v]['mcnemar_vs_V0']
            b, c = mc['b_A_acerta_B_erra'], mc['c_A_erra_B_acerta']
            dif = ab[v]['acuracia'] - a0
            # erro-padrão da diferença pareada (McNemar): sqrt(b + c − (b − c)²/n)/n
            se = 100 * np.sqrt(max(b + c - (b - c) ** 2 / n, 0)) / n
            ax = axs[0] if i < 2 else axs[1]
            y = (i % 2) + (j - 1) * 0.22
            ax.errorbar(dif, y, xerr=1.96 * se, fmt=mk, color=cor, ms=3.5, elinewidth=0.7, capsize=0,
                        mfc=cor if mc['p_valor'] < 0.05 else 'white', mew=0.8,
                        label=nome if i == 0 else None)
    for ax, lab in zip(axs, ([var[0][1], var[1][1]], [var[2][1], var[3][1]])):
        ax.axvline(0, color=CINZA, lw=0.6, zorder=0)
        ax.set_yticks([0, 1])
        ax.set_ylim(1.5, -0.5)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_yticklabels(lab)
    axs[0].set_xlim(-1.5, 1.5)
    axs[1].set_xlim(-65, 5)
    axs[1].yaxis.tick_right()
    axs[0].set_xlabel(r'$\Delta$ accuracy (pp)')
    axs[1].set_xlabel(r'$\Delta$ accuracy (pp)')
    rotulo(axs[0], 'a'); rotulo(axs[1], 'b')
    axs[0].legend(loc='upper center', bbox_to_anchor=(1.06, 1.30), ncol=3, handletextpad=0.2, columnspacing=0.8)
    salvar(fig, 'fig2_ablacao')


# ═══════════════════════════════════ Fig. 3 ═══════════════════════════════════
def perfil_rede(inst_nome='CRISP/SST', bf=30.0, seed=7701, fator=1.0):
    """Perfil V de rede (B = 300 G, f = 0,1 → B·f = 30 G) degradado pelo instrumento dado."""
    from Src.physics import INSTRUMENTAL_V2, get_stokes, degradar_stokes, reamostrar_para_LAM, V_CORE_KMS
    ins = INSTRUMENTAL_V2[inst_nome]
    st = get_stokes(300.0, filling=bf / 300.0, v_km_s=V_CORE_KMS)
    deg = degradar_stokes(st, LAM, ins, rng_local=np.random.default_rng(seed), fator_ruido=fator)
    return reamostrar_para_LAM(deg, ins)[1]


def campo_saida(m, montar, V):
    with torch.no_grad():
        x = montar(torch.tensor(V[None, :], dtype=DTYPE, device=DEVICE))
        if m.diafragma is not None:
            x = x * m.diafragma
        E = m.mascara(x, aplicar_pad=False)[0]
        ny, nx = E.shape
        E = torch.roll(E, shifts=(ny // 2, nx // 2), dims=(-2, -1))
    return x[0].cpu().numpy(), (E.abs() ** 2).cpu().numpy()


def recorte_pupila(Z):
    li, co = np.where(PUPIL > 0)
    r = slice(li.min(), li.max() + 1)
    c = slice(co.min(), co.max() + 1)
    return Z[r, c], FX[0, c], FY[r, 0]


def fig3():
    m, meta = carregar_modelo(CKPT / 'modelo_d2nn_optico_v2.pt', DEVICE)
    montar = MontadorEntrada(float(meta.get('X_scale', X_SCALE_TREINO)), DEVICE)
    V = perfil_rede(fator=0.0)                 # sem ruído: mostra a forma do pico de correlação
    X, I = campo_saida(m, montar, V)
    _, In = campo_saida(m, montar, perfil_rede(fator=1.0))
    jy0, jx0 = m.janela
    def frac_jan(J):
        a, b = J.shape[0] // 2, J.shape[1] // 2
        return J[a - jy0 // 2: a + jy0 // 2 + 1, b - jx0 // 2: b + jx0 // 2 + 1].sum() / J.sum()
    print(f'    fig3: energia na janela / total — sem ruído {frac_jan(I):.4f}, com ruído (k = 1) {frac_jan(In):.4f}')
    H = m.mascara.get_mascara().detach().cpu().numpy()
    # (−1)^j é a rampa linear que leva o pico do centro da grade à origem (o roll de
    # meia grade na leitura); removê-la mostra a máscara na convenção de pico centrado.
    ii, jj = np.indices(H.shape)
    fase = np.where(PUPIL > 0, np.angle(H * (-1.0) ** jj), np.nan)

    fig, axs = plt.subplots(1, 3, figsize=(W2, 1.85), gridspec_kw={'wspace': 0.62})
    dl = (LAM - CALIB['lambda0_core']) * 1e3                         # pm
    ny = X.shape[0]
    sel = np.abs(dl) <= 150
    im = axs[0].imshow(X[:, sel], aspect='auto', cmap='RdBu_r', origin='lower',
                       extent=[dl[sel][0], dl[sel][-1], 0.5, ny + 0.5],
                       vmin=-np.abs(X).max(), vmax=np.abs(X).max(), interpolation='nearest')
    axs[0].set_xlabel(r'$\lambda-\lambda_0$ (pm)')
    axs[0].set_ylabel('Input row $y$')
    cb = fig.colorbar(im, ax=axs[0], fraction=0.05, pad=0.03)
    cb.set_label('$X(y,\\lambda)$ (arb.)'); cb.set_ticks([]); cb.outline.set_linewidth(0.4)

    Z, fx, fy = recorte_pupila(fase)
    im = axs[1].imshow(Z, aspect='auto', cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi,
                       extent=[fx[0], fx[-1], fy[0] * 64, fy[-1] * 64], interpolation='nearest')
    axs[1].set_xlabel(r'$f_\lambda$ (cycles nm$^{-1}$)')
    axs[1].set_ylabel(r'$f_y$ (cycles per 64 rows)')
    cb = fig.colorbar(im, ax=axs[1], fraction=0.05, pad=0.03, ticks=[-np.pi, 0, np.pi])
    cb.ax.set_yticklabels([r'$-\pi$', '0', r'$\pi$']); cb.set_label('Phase (rad)'); cb.outline.set_linewidth(0.4)

    ny, nx = I.shape
    hy, hx = 8, 450
    dpm = (LAM[1] - LAM[0]) * 1e3
    Ic = I[ny // 2 - hy: ny // 2 + hy + 1, nx // 2 - hx: nx // 2 + hx + 1]
    Ic = Ic / Ic.max()
    im = axs[2].imshow(Ic, aspect='auto', cmap='magma', origin='lower', interpolation='nearest',
                       extent=[-(hx + 0.5) * dpm, (hx + 0.5) * dpm, -hy - 0.5, hy + 0.5])
    jy, jx = m.janela
    axs[2].add_patch(Rectangle((-(jx / 2) * dpm, -jy / 2), jx * dpm, jy, fill=False, ec='#56B4E9', lw=0.9))
    axs[2].set_xlabel('Output position (pm equiv.)')
    axs[2].set_ylabel('Output row offset')
    axs[2].set_yticks([-8, -4, 0, 4, 8])
    cb = fig.colorbar(im, ax=axs[2], fraction=0.05, pad=0.03, ticks=[0, 1])
    cb.set_label(r'$|E|^2$ (norm.)'); cb.outline.set_linewidth(0.4)
    for ax, l in zip(axs, 'abc'):
        rotulo(ax, l)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(True)
    frac = I[ny // 2 - jy // 2: ny // 2 + jy // 2 + 1, nx // 2 - jx // 2: nx // 2 + jx // 2 + 1].sum() / I.sum()
    print(f'    fig3: fração da energia de saída na janela 3 × 5 = {frac:.3f}; '
          f'std da fase na pupila = {np.nanstd(fase):.2f} rad')
    salvar(fig, 'fig3_pof')


# ═══════════════════════════════════ Fig. 4 ═══════════════════════════════════
def fig4():
    d75, d76 = ler('75_referencias_artigo_b'), ler('76_mascara_branqueada')
    A = d75['A_bf50']
    curvas = [('filtro |a| (LAM)', 'Unweighted matched filter', PRETO, '--', None),
              ('D²NN 67 (só fase)', 'Trained phase-only', AZUL, '-', 'o'),
              ('D²NN 74 (amp. e fase)', 'Trained amplitude + phase', VERM, '-', 's'),
              ('branqueado (instrumento)', 'Designed whitened', VERDE, '-', '^')]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 1.9), sharey=True, gridspec_kw={'wspace': 0.08})
    for ax, nm, curto, l in zip(axs, INST, INST_CURTO, 'abc'):
        k, opt = serie(A['ótimo branqueado (nativo)'][nm])
        for chave, nome, cor, ls, mk in curvas:
            fonte = d76['resultados'][nm][chave] if chave == 'branqueado (instrumento)' else A[chave][nm]
            kk, b = serie(fonte)
            assert np.allclose(kk, k)
            ax.plot(k, b / opt, ls=ls, marker=mk, color=cor, ms=3, lw=1.0 if mk else 0.9, label=nome,
                    mfc='white' if mk == '^' else cor, mew=0.8)
        # IC pareado em k = 1 das máscaras treinadas (75)
        for chave, cor, dx in (('D²NN 67 (só fase)', AZUL, 0.97), ('D²NN 74 (amp. e fase)', VERM, 1.03)):
            r = d75['A_razoes'][f'{chave} / ótimo branqueado (nativo) | {nm}']['k1']
            ax.errorbar(dx, r['razao'], yerr=[[r['razao'] - r['ic'][0]], [r['ic'][1] - r['razao']]],
                        fmt='none', ecolor=cor, elinewidth=0.8, capsize=1.5, capthick=0.6)
        eixo_k(ax)
        ax.set_title(curto, pad=2)
        rotulo(ax, l)
    axs[0].set_ylabel(r'$a_{50}$ / optimum')
    axs[0].set_ylim(0.9, 1.6)
    axs[0].legend(loc='upper center', bbox_to_anchor=(1.6, 1.32), ncol=4, handlelength=2.2, columnspacing=1.2)
    salvar(fig, 'fig4_razoes')


# ═══════════════════════════════════ Fig. 5 ═══════════════════════════════════
def fig5():
    d = ler('73_mascara_visp')['resultados']
    curvas = [('POF médio ±200 (fixo)', 'Phase-only, mixture template, ±200 pm', VERM, '--', 's', 'white'),
              ('POF ViSP sem diafr.', 'Phase-only, ViSP template, no stop', ROXO, '--', 'D', 'white'),
              ('POF ViSP ±80', 'Phase-only, ViSP template, ±80 pm', ROXO, '-', 'D', None),
              ('67 (atual)', 'Phase-only, trained (±200 pm)', AZUL, '-', 'o', None),
              ('casado complexo ViSP sem diafr.', r'Amplitude + phase, $H=S^*$ (ViSP)', VERDE, '-', '^', None)]
    fig, ax = plt.subplots(figsize=(W1, 2.15))
    for chave, nome, cor, ls, mk, mf in curvas:
        k = np.array([e['k'] for e in d[chave]])
        r = np.array([e['razao'] for e in d[chave]])
        ax.plot(k, r, ls=ls, marker=mk, color=cor, ms=3, lw=1.0, label=nome, mfc=mf or cor, mew=0.8)
    eixo_k(ax)
    ax.set_ylabel(r'$a_{50}$ / matched filter (ViSP)')
    ax.set_ylim(0.8, 2.4)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.42), ncol=1, handlelength=2.4, labelspacing=0.25)
    salvar(fig, 'fig5_visp')


# ═══════════════════════════════════ Fig. 6 ═══════════════════════════════════
def mascara_branqueada_crisp():
    """Máscara projetada H = conj F{X_h}/max (script 76), CRISP."""
    spec = importlib.util.spec_from_file_location('s76', ROOT / 'Scripts' / '76_mascara_branqueada.py')
    s76 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s76)
    m67, meta = carregar_modelo(CKPT / 'modelo_d2nn_optico_v2.pt', DEVICE)
    montar = MontadorEntrada(float(meta.get('X_scale', X_SCALE_TREINO)), DEVICE)
    w = s76.pesos_branqueados_lam()
    m, _ = s76.mascara(montar, w[1])
    return m.mascara.get_mascara().detach().cpu().numpy()


def fig6():
    m74, _ = carregar_modelo(CKPT / 'modelo_d2nn_complexo_v2.pt', DEVICE)
    H74 = m74.mascara.get_mascara().detach().cpu().numpy()
    Hw = mascara_branqueada_crisp()
    d75, d76 = ler('75_referencias_artigo_b'), ler('76_mascara_branqueada')

    fig = plt.figure(figsize=(W2, 1.95))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 1.25], wspace=0.3)
    axs = [fig.add_subplot(gs[0, i]) for i in (0, 1, 3)]
    cax = fig.add_subplot(gs[0, 2])
    for ax, H, l in ((axs[0], H74, 'a'), (axs[1], Hw, 'b')):
        a = np.where(PUPIL > 0, np.abs(H), np.nan)
        Z, fx, fy = recorte_pupila(a)
        Z = Z / np.nanmax(Z)
        im = ax.imshow(Z, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=1,
                       extent=[fx[0], fx[-1], fy[0] * 64, fy[-1] * 64], interpolation='nearest')
        ax.set_xlabel(r'$f_\lambda$ (cycles nm$^{-1}$)')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(True)
        rotulo(ax, l)
    axs[0].set_ylabel(r'$f_y$ (cycles per 64 rows)')
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
    cb.set_label(r'$|H|$ / max'); cb.outline.set_linewidth(0.4)

    axs[1].set_yticklabels([])
    ax = axs[2]
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.06, pos.y0, pos.width - 0.06, pos.height])
    nm = 'CRISP/SST'
    A, R = d75['A_bf50'], d76['resultados'][nm]
    k, opt = serie(A['ótimo branqueado (nativo)'][nm])
    curvas = [(A['filtro |a| (LAM)'][nm], 'Unweighted matched filter', PRETO, '--', None, None),
              (A['D²NN 67 (só fase)'][nm], 'Trained phase-only', AZUL, '-', 'o', None),
              (A['D²NN 74 (amp. e fase)'][nm], 'Trained amplitude + phase', VERM, '-', 's', None),
              (R['branqueado (instrumento)'], 'Designed whitened (CRISP)', VERDE, '-', '^', 'white'),
              (R['branqueado único (3 instr.)'], 'Designed whitened (single)', VERDE, ':', 'v', 'white')]
    for fonte, nome, cor, ls, mk, mf in curvas:
        kk, b = serie(fonte)
        assert np.allclose(kk, k)
        ax.plot(k, b / opt, ls=ls, marker=mk, color=cor, ms=3, lw=1.0, label=nome, mfc=mf or cor, mew=0.8)
    eixo_k(ax)
    ax.set_ylabel(r'$a_{50}$ / optimum (CRISP)')
    ax.set_ylim(0.9, 1.45)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.02), handlelength=2.2, labelspacing=0.3)
    rotulo(ax, 'c')
    pup = PUPIL > 0
    print(f'    fig6: <|H|²> treinada 74 = {np.mean(np.abs(H74[pup]) ** 2):.4f}, '
          f'projetada CRISP = {np.mean(np.abs(Hw[pup]) ** 2):.4f}')
    salvar(fig, 'fig6_complexa')


# ═══════════════════════════════════ Fig. 1 ═══════════════════════════════════
def fig1():
    """Esquema: (a) codificação X(y,λ) = c_y V(λ); (b) processador 4f com uma máscara e um detector."""
    from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
    V = perfil_rede('ViSP/DKIST', fator=0.0)
    dl = (LAM - CALIB['lambda0_core']) * 1e3
    sel = np.abs(dl) <= 150
    a = np.linspace(0.8, 1.2, 64)
    c = a - a.mean()

    fig = plt.figure(figsize=(W2, 2.05))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 0.22, 0.8, 3.6], height_ratios=[1, 1],
                          wspace=0.35, hspace=0.9)
    axV = fig.add_subplot(gs[0, 0])
    axC = fig.add_subplot(gs[:, 1])
    axX = fig.add_subplot(gs[:, 2])
    axO = fig.add_subplot(gs[:, 3])
    axT = fig.add_subplot(gs[1, 0]); axT.axis('off')

    axV.plot(dl[sel], V[sel] / np.abs(V).max(), color=AZUL, lw=1.0)
    axV.axhline(0, color=CINZA, lw=0.4)
    axV.set_xlabel(r'$\lambda-\lambda_0$ (pm)', labelpad=1)
    axV.set_yticks([]); axV.set_xticks([-100, 0, 100])
    axV.spines['left'].set_visible(False)
    axV.set_title('Measured $V(\\lambda)$', pad=2)
    rotulo(axV, 'a')
    axT.text(0.5, 0.75, r'$X(y,\lambda)=c_y\,V(\lambda)$', ha='center', va='center', transform=axT.transAxes)
    axT.text(0.5, 0.25, r'$c_y\propto a_y-\langle a\rangle$, $a_y\in[0.8,1.2]$', ha='center', va='center',
             transform=axT.transAxes, fontsize=7, color='#444444')

    axC.barh(np.arange(64), c, height=1.0, color=np.where(c > 0, VERM, AZUL), lw=0)
    axC.set_ylim(-0.5, 63.5); axC.set_xticks([]); axC.set_yticks([0, 63]); axC.set_yticklabels(['1', '64'])
    axC.axvline(0, color=CINZA, lw=0.4)
    axC.spines['bottom'].set_visible(False)
    axC.set_title('$c_y$', pad=2); axC.set_ylabel('Row $y$', labelpad=0)

    X = c[:, None] * V[None, sel]
    axX.imshow(X, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-np.abs(X).max(), vmax=np.abs(X).max(),
               extent=[dl[sel][0], dl[sel][-1], 0.5, 64.5], interpolation='nearest')
    axX.set_yticks([]); axX.set_xticks([-100, 0, 100])
    axX.set_xlabel(r'$\lambda-\lambda_0$ (pm)', labelpad=1)
    axX.set_title('Input field $X$', pad=2)
    for s_ in ('top', 'right'):
        axX.spines[s_].set_visible(True)

    # ── (b) processador 4f ──
    ax = axO
    ax.set_xlim(-0.9, 10.9); ax.set_ylim(-1.55, 1.5); ax.axis('off')
    rotulo(ax, 'b')
    ax.plot([0.2, 9.0], [0, 0], color=CINZA, lw=0.4, ls=(0, (4, 3)), zorder=0)
    raio = dict(color='#F0B000', alpha=0.25, lw=0)
    xs = [0.7, 2.15, 3.6, 5.05, 6.5]
    # feixe: colimado até L1, converge no plano de Fourier, diverge até L2, converge na saída
    ax.fill([xs[0], xs[1], xs[2], xs[1], xs[0]], [0.85, 0.85, 0.0, -0.85, -0.85], **raio)
    ax.fill([xs[2], xs[3], xs[3], xs[2]], [0.0, 0.85, -0.85, 0.0], **raio)
    ax.fill([xs[3], xs[4], xs[3]], [0.85, 0.0, -0.85], **raio)
    for x in (xs[1], xs[3]):
        ax.add_patch(Ellipse((x, 0), 0.22, 1.95, fc='#DCEBF5', ec=AZUL, lw=0.8))
    ax.text(xs[1], -1.15, '$L_1$', ha='center'); ax.text(xs[3], -1.15, '$L_2$', ha='center')
    # plano de entrada: modulador + diafragma
    ax.add_patch(Rectangle((xs[0] - 0.09, -0.85), 0.18, 1.7, fc='#E8E8E8', ec=PRETO, lw=0.7))
    ax.plot([xs[0] + 0.16] * 2, [0.92, 1.1], color=PRETO, lw=1.4)
    ax.plot([xs[0] + 0.16] * 2, [-0.92, -1.1], color=PRETO, lw=1.4)
    ax.text(xs[0], 1.18, 'Input SLM\n+ field stop', ha='center', va='bottom', fontsize=7)
    ax.text(xs[0] - 0.5, 0.06, 'Coherent\nbeam', ha='right', va='center', fontsize=7)
    ax.annotate('', xy=(xs[0] - 0.12, 0), xytext=(xs[0] - 0.42, 0),
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color=PRETO))
    # máscara de Fourier
    ax.add_patch(Rectangle((xs[2] - 0.09, -0.85), 0.18, 1.7, fc='#F6D7C3', ec=VERM, lw=0.8))
    ax.text(xs[2], 1.18, 'Fourier\nmask $H$', ha='center', va='bottom', fontsize=7)
    # plano de saída + janela
    ax.add_patch(Rectangle((xs[4] - 0.06, -0.85), 0.12, 1.7, fc='white', ec=PRETO, lw=0.7))
    ax.add_patch(Rectangle((xs[4] - 0.09, -0.12), 0.18, 0.24, fc='#56B4E9', ec=PRETO, lw=0.6))
    ax.text(xs[4], 1.18, 'Output,\n3 × 5 window', ha='center', va='bottom', fontsize=7)
    # distâncias f
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.annotate('', xy=(x1, -1.42), xytext=(x0, -1.42),
                    arrowprops=dict(arrowstyle='<->', lw=0.5, color='#555555', shrinkA=0, shrinkB=0))
        ax.text((x0 + x1) / 2, -1.38, '$f$', ha='center', va='bottom', fontsize=7, color='#555555')
    # eletrônica
    ax.annotate('', xy=(7.45, 0), xytext=(xs[4] + 0.12, 0),
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color=PRETO))
    ax.text((xs[4] + 7.45) / 2, 0.1, '$E_W$', ha='center', va='bottom', fontsize=7)
    ax.add_patch(FancyBboxPatch((7.5, -0.6), 3.3, 1.2, boxstyle='round,pad=0.03,rounding_size=0.12',
                                fc='#F2F2F2', ec=PRETO, lw=0.7))
    ax.text(9.15, 0.22, r'$z=(\ln E_W-\mu)/s$', ha='center', va='center', fontsize=7)
    ax.text(9.15, -0.22, r'$w_k z+b_k$, $k=1,2,3$', ha='center', va='center', fontsize=7)
    ax.text(9.15, -0.68, '6 parameters →\nquiet / network / spot', ha='center', va='top', fontsize=7)
    salvar(fig, 'fig1_esquema')


# ═══════════════════════════════════ Fig. S1 ═══════════════════════════════════
def figS1():
    """Suplemento: a50(Δv)/a50(0) de cada detector (script 68; razões, independentes do estimador)."""
    d = ler('68_doppler')
    R = d['resultados']
    dvs = sorted(R, key=float)
    curvas = [('67', 'Trained phase-only, one window', AZUL, '-', 'o', None),
              ('67 + fileira', 'Trained phase-only, row of 19 windows', AZUL, ':', 'o', 'white'),
              ('filtro |a|', 'Matched filter, fixed template', PRETO, '--', None, None),
              ('banco |a|', 'Matched filter, template bank', VERM, '-', 's', 'white')]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 1.9), sharey=True, gridspec_kw={'wspace': 0.08})
    for ax, nm, curto, l in zip(axs, INST, INST_CURTO, 'abc'):
        for chave, nome, cor, ls, mk, mf in curvas:
            x = np.array([float(v) for v in dvs])
            y = np.array([R[v][chave]['bf50'][nm] / R['0.0'][chave]['bf50'][nm] for v in dvs])
            ax.plot(x, y, ls=ls, marker=mk or 'none', color=cor, ms=3, lw=1.0, label=nome, mfc=mf or cor, mew=0.8)
        ax.axhline(1.0, color=CINZA, lw=0.6, zorder=0)
        ax.set_xlabel(r'Doppler shift $\Delta v$ (km s$^{-1}$)')
        ax.set_xticks([-2, 0, 2, 4])
        ax.set_title(curto, pad=2)
        rotulo(ax, l)
    axs[0].set_ylabel(r'$a_{50}(\Delta v)\,/\,a_{50}(0)$')
    axs[0].set_ylim(0.9, 1.15)
    axs[0].legend(loc='upper center', bbox_to_anchor=(1.6, 1.34), ncol=2, handlelength=2.4, columnspacing=1.5)
    salvar(fig, 'figS1_doppler')


if __name__ == '__main__':
    print(f'Figuras em {OUT}')
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, figS1):
        f()
