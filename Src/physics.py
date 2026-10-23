"""
physics.py — Constantes físicas e de calibração do problema (linha Ca II 854.2 nm)

Fonte única da verdade para os parâmetros físicos e instrumentais.
Qualquer alteração de calibração deve ser feita APENAS aqui.

Corresponde à Célula 1 do notebook original.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d 
from scipy.interpolate import interp1d
# -----------------------------------------------------------------------
# CALIB — constantes físicas solares puras (sem parâmetros instrumentais)
# -----------------------------------------------------------------------
CALIB = {
    # Constantes físicas da linha Ca II 854.2 nm
    'lambda0'      : 854.2,
    'lambda0_core' : 854.2118,
    'g_eff'        : 1.10,
    'c_zee'        : 4.6686e-12,
    'c_light'      : 299792.0,

    # Modelo de dois componentes atmosféricos
    'sigma_f'  : 0.245,      # [nm] = 2,45 Å — recalibrado: FWHM = 253,0 pm = 2,53 Å (err=0.007%)
                             # (o comentário antigo dizia "245 mÅ": o rótulo mÅ era pm; ver DL_mA)
    'gamma_f'  : 0.100,
    'sigma_c'  : 0.045,
    'gamma_c'  : 0.030,
    'weight_c' : 0.55,
    'depth'    : 0.812,

    # Grade espectral
    'n_lam'    : 3000,
    'lam_min'  : 853.7,
    'lam_max'  : 854.7,

    # Métricas do atlas FTS/Neckel
    'atlas_I_min' : 0.18801,
    'atlas_FWHM'  : 253.0,   # [pm] (= 2,53 Å); rótulo histórico "mÅ" estava errado
    'atlas_asym'  : -26.6,
}

# -----------------------------------------------------------------------
# INSTRUMENTAL — modelo instrumental LEGADO (scripts 04–09 e 46–54)
#
# ATENÇÃO (achado 16, docs/MODELO_INSTRUMENTAL.md): no modo legado,
# degradar_perfil/degradar_stokes convertem 'fwhm_mA' e 'passo_mA' com ÷1000,
# como se fossem pm. Os valores nominais estão em mÅ (1 mÅ = 10⁻⁴ nm), então a
# LSF e o passo APLICADOS são 10× maiores (IBIS: 430 mÅ em vez de 43 mÅ) e a
# grade cobre a janela inteira (±5 Å). Mantido sem alteração para reproduzir
# bit a bit os resultados antigos. Para trabalho novo, usar INSTRUMENTAL_V2.
# -----------------------------------------------------------------------
INSTRUMENTAL = {
    'IBIS/DST': {
        'modelo'      : 'legado',
        'fwhm_mA'     : 43.0,
        'passo_mA'    : 30.0,
        'snr_foton'   : 1000.0,
        'ron_e'       : 5.0,
        'stray_light' : 0.05,    # Cauzzi et al. (2008)
        'cor'         : '#79c0ff',
    },
    'CRISP/SST': {
        'modelo'      : 'legado',
        'fwhm_mA'     : 65.0,
        'passo_mA'    : 65.0,
        'snr_foton'   : 500.0,
        'ron_e'       : 3.0,
        'stray_light' : 0.06,    # Scharmer et al. (2008)
        'cor'         : '#ffa657',
    },
    'ViSP/DKIST': {
        'modelo'      : 'legado',
        'fwhm_mA'     : 30.0,
        'passo_mA'    : 20.0,
        'snr_foton'   : 2000.0,
        'ron_e'       : 2.0,
        'stray_light' : 0.02,
        'cor'         : '#56d364',
    },
}
INSTRUMENTAL_LEGADO = INSTRUMENTAL

# -----------------------------------------------------------------------
# INSTRUMENTAL_V2 — unidades corretas, LSF e grades da literatura, ruído do
# CRISP medido nos dados do SST (script 55). Grades em pm relativas ao núcleo
# da linha em repouso (CALIB['lambda0_core']).
# -----------------------------------------------------------------------
# IBIS: grade representativa de Cauzzi et al. (2008, Tabela 1: 27–35 pontos,
# passos de 4–16 pm, de −120 a +160 pm).
GRADE_IBIS_PM = np.arange(-120.0, 160.0 + 1e-9, 10.0)
# CRISP: PROVISÓRIA — 15 pontos em ±90 pm (programas publicados de 15 pontos:
# ±90 pm, ou −60 a +75 pm). Substituir pelos λ reais do arquivo de 2024-07-08
# (tabela WCS-TAB ausente na exportação; ver script 55).
GRADE_CRISP_PM_PROVISORIA = np.array([-90.0, -70.0, -50.0, -35.0, -25.0, -15.0, -7.0, 0.0,
                                      7.0, 15.0, 25.0, 35.0, 50.0, 70.0, 90.0])

INSTRUMENTAL_V2 = {
    'IBIS/DST': {
        'modelo'      : 'v2',
        'fwhm_pm'     : 4.4,     # Reardon & Cavallini, citado em Cauzzi et al. (2008)
        'grade_dl_pm' : GRADE_IBIS_PM,
        'stray_light' : 0.05,
        'ruido'       : {'tipo': 'constante', 'sigma': 1.0e-3},     # NÃO medido (valor antigo)
        'cor'         : '#79c0ff',
    },
    'CRISP/SST': {
        'modelo'      : 'v2',
        'fwhm_pm'     : 10.5,    # perfil de transmissão do CRISP em 854,2 nm (literatura)
        'grade_dl_pm' : GRADE_CRISP_PM_PROVISORIA,
        'grade_provisoria': True,
        'stray_light' : 0.06,
        # Medido no SST (script 55): σ_V ≈ 0,32% da intensidade LOCAL por varredura,
        # σ ∝ I^0,93 (não é ruído de fótons), independente entre varreduras (média de 10:
        # ganho 3,11; ideal 3,16), correlacionado no espaço (2×2: ganho 1,11; ideal 2).
        'ruido'       : {'tipo': 'proporcional_I', 'eps': 3.2e-3},
        'ganhos_medidos': {'media_10_varreduras': 3.11, 'agrupamento_2x2': 1.11,
                           'agrupamento_4x4': 1.61},
        'cor'         : '#ffa657',
    },
    'ViSP/DKIST': {
        'modelo'      : 'v2',
        'fwhm_pm'     : 4.7,     # R ≳ 180 000 (de Wijn et al. 2022) → 854,2 nm / 180 000
        'passo_pm'    : 2.4,     # amostragem crítica (≈ FWHM/2), janela inteira
        'stray_light' : 0.02,
        'ruido'       : {'tipo': 'constante', 'sigma': 0.5e-3},     # NÃO medido (valor antigo)
        'cor'         : '#56d364',
    },
}

# Instrumento de referência para diagnósticos que precisam de um valor padrão
INST_REF = INSTRUMENTAL['IBIS/DST']

# -----------------------------------------------------------------------
# Grade espectral
# -----------------------------------------------------------------------
LAM   = np.linspace(CALIB['lam_min'], CALIB['lam_max'],
                    CALIB['n_lam'], dtype=np.float64)
DL    = LAM - CALIB['lambda0']
DL_pm = DL * 1000          # [pm]
DL_mA = DL_pm              # NOME HISTÓRICO: o valor está em pm (1 pm = 10 mÅ), não em mÅ.
                           # Mantido para não quebrar os scripts 03–19; usar DL_pm em código novo.
DLAM  = LAM[1] - LAM[0]

V_CORE_KMS = (CALIB['c_light']
              * (CALIB['lambda0_core'] - CALIB['lambda0'])
              / CALIB['lambda0'])

# -----------------------------------------------------------------------
# Verificação de integridade
# -----------------------------------------------------------------------
assert 'stray_light' not in CALIB, \
    'C3 VIOLADA: stray_light em CALIB. Mover para INSTRUMENTAL.'
assert 'fwhm_ibis' not in CALIB, \
    'C3 VIOLADA: fwhm_ibis em CALIB. Mover para INSTRUMENTAL.'


# -----------------------------------------------------------------------
# Motor Físico Canônico: pseudo-Voigt + Stokes Completo
# Corresponde à Célula 2 do notebook original.
# -----------------------------------------------------------------------

def _pseudo_voigt(x, sigma, gamma):
    """Perfil pseudo-Voigt normalizado (Thompson, Cox & Hastings 1987)."""
    fG  = 2.0 * sigma * np.sqrt(2.0 * np.log(2.0))
    fL  = 2.0 * gamma
    f   = (fG**5
           + 2.69269*fG**4*fL + 2.42843*fG**3*fL**2
           + 4.47163*fG**2*fL**3 + 0.07842*fG*fL**4
           + fL**5) ** 0.2
    eta = (1.36603*(fL/f) - 0.47719*(fL/f)**2 + 0.11116*(fL/f)**3)
    G   = np.exp(-4.0*np.log(2.0)*x**2/f**2)
    L   = 1.0 / (1.0 + 4.0*x**2/f**2)
    return (eta*L + (1.0-eta)*G).astype(np.float64)


def get_stokes(B, theta_deg=0.0, chi_deg=0.0, filling=1.0,
               v_km_s=0.0, lam_grid=None, preench_nucleo=0.0,
               contraste_cont=1.0):
    """
    Motor físico canônico — perfil solar PURO (sem stray_light).

    Para observações simuladas, passar o resultado por degradar_stokes()
    com o instrumento desejado.

    Termo térmico (só no componente MAGNÉTICO, peso `filling`):
      preench_nucleo η : fração do componente cromosférico estreito que é
                         "preenchida" — o peso do núcleo passa de wc para
                         wc·(1 − η). η = 0: Sol quieto; η > 0: núcleo mais
                         brilhante (plage/fácula); η > 1 inverte o núcleo
                         (emissão, regime de flare — fora do escopo).
      contraste_cont κ : intensidade do contínuo do componente magnético
                         em unidades do contínuo quieto. κ = 1: quieto;
                         κ < 1: umbra/penumbra; κ ≳ 1: fácula.
                         I, V, Q e U do componente magnético escalam com κ
                         (a polarização vem da mesma luz, mais fraca).
    Com os padrões (η = 0, κ = 1) o resultado é idêntico ao modelo
    validado no Bloco 1 (atlas FTS/Neckel, Cauzzi).

    Retorna dict {'I', 'V', 'Q', 'U'} em unidades de Ic (contínuo quieto).
    """
    if lam_grid is None:
        lam_grid = LAM
    dl_g = lam_grid - CALIB['lambda0']

    L0    = CALIB['lambda0']
    dB    = CALIB['c_zee'] * CALIB['g_eff'] * B * L0**2
    th    = np.radians(theta_deg)
    ch    = np.radians(chi_deg)
    depth = CALIB['depth']
    wc    = CALIB['weight_c']

    dl_dop = L0 * (v_km_s / CALIB['c_light'])
    x      = dl_g - dl_dop

    def _pv_pair(sig, gam):
        return (_pseudo_voigt(x+dB, sig, gam),
                _pseudo_voigt(x-dB, sig, gam),
                _pseudo_voigt(x,    sig, gam))

    fp, fm, f0 = _pv_pair(CALIB['sigma_f'], CALIB['gamma_f'])
    cp, cm, c0 = _pv_pair(CALIB['sigma_c'], CALIB['gamma_c'])

    def _mix(af, ac, peso_c=wc):
        return (1.0-wc)*af + peso_c*ac

    # Componente magnético: núcleo cromosférico com peso wc·(1 − η)
    wc_mag = wc * (1.0 - preench_nucleo)
    phi_p = _mix(fp, cp, wc_mag)
    phi_m = _mix(fm, cm, wc_mag)
    phi_0 = _mix(f0, c0, wc_mag)

    cos_th  = np.cos(th)
    sin2_th = np.sin(th)**2
    cos2_ch = np.cos(2.0*ch)
    sin2_ch = np.sin(2.0*ch)

    w_sig_I = 0.25*(1.0 + cos_th**2)
    w_pi_I  = 0.50*sin2_th
    norm_I  = 2.0*w_sig_I + w_pi_I

    abs_I     = (w_sig_I*(phi_p+phi_m) + w_pi_I*phi_0) / norm_I
    phi_quiet = _mix(f0, c0)
    I_mag     = 1.0 - depth*abs_I
    I_quiet   = 1.0 - depth*phi_quiet

    V_profile = 0.5  * depth * (phi_p-phi_m) * cos_th
    L_profile = 0.25 * depth * sin2_th * (phi_p - 2.0*phi_0 + phi_m)

    # Filling factor — sem stray_light (C3); κ escala o componente magnético
    k = contraste_cont
    I_out = filling*k*I_mag  + (1.0-filling)*I_quiet
    V_out = filling*k*V_profile
    Q_out = filling*k*L_profile*cos2_ch
    U_out = filling*k*L_profile*sin2_ch

    return {
        'I': I_out.astype(np.float64),
        'V': V_out.astype(np.float64),
        'Q': Q_out.astype(np.float64),
        'U': U_out.astype(np.float64),
    }


def get_stokes_IQ(B, filling=1.0, v_km_s=0.0, lam_grid=None):
    return get_stokes(B, theta_deg=0.0, filling=filling,
                      v_km_s=v_km_s, lam_grid=lam_grid)


# -----------------------------------------------------------------------
# Asserts de consistência física (rodam sempre que este módulo é importado
# — funcionam como trava de segurança contra regressões na física)
# -----------------------------------------------------------------------
from scipy.interpolate import UnivariateSpline

_st0 = get_stokes(B=0)
assert np.allclose(_st0['V'], 0.0, atol=1e-12), 'ERRO: V ≠ 0 para B=0'

I_model   = get_stokes(B=0, v_km_s=V_CORE_KMS)['I']
I_min_mod = I_model.min()
err_Imin  = abs(I_min_mod - CALIB['atlas_I_min']) / CALIB['atlas_I_min'] * 100

assert err_Imin < 5.0, \
    f'C3: I_min={I_min_mod:.5f}, erro={err_Imin:.2f}% > 5%'

_depth_mod = 1.0 - I_min_mod
_spl       = UnivariateSpline(DL_mA, I_model-(1.0-_depth_mod/2.0), s=0)
_roots     = _spl.roots()
FWHM_SOLAR = float(_roots[-1]-_roots[0]) if len(_roots) >= 2 else float('nan')
FWHM_ROOTS_mA = _roots
err_fwhm   = abs(FWHM_SOLAR-CALIB['atlas_FWHM'])/CALIB['atlas_FWHM']*100

assert 220 < FWHM_SOLAR < 265, \
    f'ERRO: FWHM={FWHM_SOLAR:.1f} pm fora da faixa solar'

_V_amps = [np.ptp(get_stokes(b)['V']) for b in [100, 500, 1000]]
assert _V_amps[0] < _V_amps[1] < _V_amps[2], \
    'ERRO: V não cresce monotonicamente com B'

st_val  = get_stokes(B=0, filling=1.0)
assert np.allclose(st_val['V'], 0.0, atol=1e-12)

# -----------------------------------------------------------------------
# Diagnóstico — só roda se você executar este arquivo diretamente
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print('✓ Setup concluído.')
    print(f'  Grade   : {CALIB["n_lam"]} pts | '
          f'{CALIB["lam_min"]:.1f}–{CALIB["lam_max"]:.1f} nm | '
          f'Passo: {DLAM*1000:.3f} pm')
    print(f'  v_núcleo: +{V_CORE_KMS:.3f} km/s')
    print(f'  CALIB        → física solar pura (sem stray_light)  ✓')
    print(f'  INSTRUMENTAL → 3 instrumentos com stray_light próprio  ✓')

    print('\n✓ Motor canônico carregado.')
    print(f'{"="*56}')
    print(f'  I_min  : {I_min_mod:.5f} Ic  | Atlas: {CALIB["atlas_I_min"]:.5f} '
          f'| Erro: {err_Imin:.3f}%  ✓')
    print(f'  FWHM   : {FWHM_SOLAR:.1f} pm  | Atlas: {CALIB["atlas_FWHM"]:.0f} pm '
          f'| Erro: {err_fwhm:.3f}%  ✓')
    print(f'  V monot.: ✓  |  stray_light: NÃO aplicado aqui  ✓')
    print(f'{"="*56}')
    
# -----------------------------------------------------------------------
# Degradação Instrumental: LSF + Passo Espectral + Ruído
# Corresponde à Célula 3B do notebook original.
#
# stray_light é aplicado APENAS aqui (nunca em get_stokes()).
# Ordem: stray → LSF → reamostramento → ruído  (Mathew et al. 2009)
# -----------------------------------------------------------------------

def _sigma_ruido_legado(instrumento):
    """Desvio padrão do ruído (fótons + leitura) do instrumento, em unidades de Ic."""
    sigma_fot = 1.0 / instrumento['snr_foton']
    sigma_ron = instrumento['ron_e'] / instrumento['snr_foton']**2
    return np.sqrt(sigma_fot**2 + sigma_ron**2)


def _degradar_perfil_legado(perfil, lam_grid_nm, instrumento, rng_local=None, fator_ruido=1.0):
    """
    Cadeia de degradação para um único perfil (Stokes I).

    Etapas: stray (0) → LSF (1) → reamostramento (2) → ruído (3).
    Para degradar o vetor Stokes completo usar degradar_stokes().

    fator_ruido : multiplica o ruído (1 = instrumento real; 0 = sem ruído, mantendo
                  luz espúria, LSF e amostragem). O sorteio aleatório é o mesmo para
                  qualquer fator (o ruído é sorteado e depois escalado), então a
                  sequência do gerador não muda; com 1 o resultado é idêntico ao
                  anterior à introdução deste parâmetro.

    Retorna: lam_obs, obs (float32), sigma_tot (desvio padrão efetivo = fator × σ)
    """
    if rng_local is None:
        rng_local = np.random.default_rng()

    perfil_f64 = perfil.astype(np.float64)
    dlam_nm    = lam_grid_nm[1] - lam_grid_nm[0]

    # Etapa 0: stray light
    alpha        = instrumento.get('stray_light', 0.0)
    perfil_stray = (1.0 - alpha) * perfil_f64 + alpha * 1.0

    # Etapa 1: LSF
    fwhm_nm    = instrumento['fwhm_mA'] / 1000.0
    sigma_px   = fwhm_nm / dlam_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    perfil_lsf = gaussian_filter1d(perfil_stray, sigma=sigma_px)

    # Etapa 2: reamostramento
    passo_nm  = instrumento['passo_mA'] / 1000.0
    lam_obs   = np.arange(lam_grid_nm[0], lam_grid_nm[-1], passo_nm)
    obs_clean = interp1d(lam_grid_nm, perfil_lsf,
                         kind='linear', fill_value=1.0,
                         bounds_error=False)(lam_obs)

    # Etapa 3: ruído
    sigma_tot = _sigma_ruido_legado(instrumento)

    ruido = rng_local.normal(0.0, sigma_tot, len(obs_clean)) * fator_ruido
    obs   = (obs_clean + ruido).astype(np.float32)

    return lam_obs, obs, sigma_tot * fator_ruido


def _degradar_stokes_legado(stokes_dict, lam_grid_nm, instrumento, rng_local=None,
                            fator_ruido=1.0):
    """
    Aplica degradação completa ao vetor de Stokes {I, V, Q, U}.

    Fisicamente correto:
      I → (1-α)·I + α·1.0  via degradar_perfil() (etapas 0–3)
      V, Q, U → (1-α)·X + LSF + reamostramento + ruído
                (sem fundo — luz espalhada não é polarizada)

    fator_ruido : ver degradar_perfil (1 = real, 0 = sem ruído).

    Retorna dict com 'I','V','Q','U','lam_obs','sigma_tot','alpha'.
    """
    if rng_local is None:
        rng_local = np.random.default_rng()

    alpha = instrumento.get('stray_light', 0.0)
    out   = {}

    lam_obs, I_obs, sigma_ef = _degradar_perfil_legado(
        stokes_dict['I'], lam_grid_nm, instrumento, rng_local, fator_ruido)
    sigma_tot = _sigma_ruido_legado(instrumento)
    out['I'] = I_obs

    dlam_nm  = lam_grid_nm[1] - lam_grid_nm[0]
    fwhm_nm  = instrumento['fwhm_mA'] / 1000.0
    sigma_px = fwhm_nm / dlam_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    passo_nm = instrumento['passo_mA'] / 1000.0

    for key in ['V', 'Q', 'U']:
        p_stray = stokes_dict[key].astype(np.float64) * (1.0 - alpha)
        p_lsf   = gaussian_filter1d(p_stray, sigma=sigma_px)
        lam_p   = np.arange(lam_grid_nm[0], lam_grid_nm[-1], passo_nm)
        p_clean = interp1d(lam_grid_nm, p_lsf,
                           kind='linear', fill_value=0.0,
                           bounds_error=False)(lam_p)
        ruido    = rng_local.normal(0.0, sigma_tot, len(p_clean)) * fator_ruido
        out[key] = (p_clean + ruido).astype(np.float32)

    out['lam_obs']   = lam_obs
    out['sigma_tot'] = sigma_ef
    out['alpha']     = alpha
    return out



# -----------------------------------------------------------------------
# Modelo instrumental v2 (unidades corretas) e funções públicas
# -----------------------------------------------------------------------
def _eh_legado(instrumento):
    return instrumento.get('modelo', 'legado') == 'legado'


def _lam_obs_v2(instrumento, lam_grid_nm):
    """λ amostrados: grade explícita (pm relativos ao núcleo em repouso) ou passo na janela."""
    if instrumento.get('grade_dl_pm') is not None:
        return CALIB['lambda0_core'] + np.asarray(instrumento['grade_dl_pm'], dtype=np.float64) * 1e-3
    return np.arange(lam_grid_nm[0], lam_grid_nm[-1], instrumento['passo_pm'] * 1e-3)


def _sigma_por_ponto_v2(instrumento, I_limpo):
    r = instrumento['ruido']
    if r['tipo'] == 'constante':
        return np.full(len(I_limpo), float(r['sigma']))
    if r['tipo'] == 'proporcional_I':
        return float(r['eps']) * np.abs(np.asarray(I_limpo, dtype=np.float64))
    raise ValueError(f'tipo de ruído desconhecido: {r["tipo"]}')


def _degradar_stokes_v2(stokes_dict, lam_grid_nm, instrumento, rng_local, fator_ruido):
    alpha = instrumento.get('stray_light', 0.0)
    dlam = lam_grid_nm[1] - lam_grid_nm[0]
    sig_px = instrumento['fwhm_pm'] * 1e-3 / dlam / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    lam_obs = _lam_obs_v2(instrumento, lam_grid_nm)

    def amostrar(p):
        return np.interp(lam_obs, lam_grid_nm, gaussian_filter1d(p, sigma=sig_px))

    I_limpo = amostrar((1.0 - alpha) * stokes_dict['I'].astype(np.float64) + alpha)
    sig = _sigma_por_ponto_v2(instrumento, I_limpo)
    out = {'I': (I_limpo + rng_local.normal(0.0, 1.0, len(lam_obs)) * sig * fator_ruido).astype(np.float32)}
    for key in ('V', 'Q', 'U'):
        limpo = amostrar(stokes_dict[key].astype(np.float64) * (1.0 - alpha))
        out[key] = (limpo + rng_local.normal(0.0, 1.0, len(lam_obs)) * sig * fator_ruido).astype(np.float32)
    out['lam_obs'] = lam_obs
    out['sigma_por_ponto'] = sig * fator_ruido
    out['sigma_tot'] = sigma_ruido(instrumento) * fator_ruido
    out['alpha'] = alpha
    return out


def sigma_ruido(instrumento):
    """
    σ de referência do instrumento em unidades de Ic.
    Legado: fótons + leitura. v2 constante: 'sigma'. v2 proporcional a I: 'eps'
    (σ num ponto com I = Ic; em geral σ_k = eps·I_k).
    """
    if _eh_legado(instrumento):
        return _sigma_ruido_legado(instrumento)
    r = instrumento['ruido']
    return float(r['sigma'] if r['tipo'] == 'constante' else r['eps'])


def degradar_perfil(perfil, lam_grid_nm, instrumento, rng_local=None, fator_ruido=1.0):
    """
    Degrada um perfil (Stokes I). Legado: _degradar_perfil_legado (bit a bit
    idêntico à versão anterior). v2: luz espúria → LSF (fwhm_pm) → amostragem na
    grade do instrumento → ruído do modelo v2.
    Retorna: lam_obs, obs (float32), σ de referência × fator.
    """
    if rng_local is None:
        rng_local = np.random.default_rng()
    if _eh_legado(instrumento):
        return _degradar_perfil_legado(perfil, lam_grid_nm, instrumento, rng_local, fator_ruido)
    z = np.zeros_like(perfil, dtype=np.float64)
    out = _degradar_stokes_v2({'I': perfil, 'V': z, 'Q': z, 'U': z}, lam_grid_nm, instrumento,
                              rng_local, fator_ruido)
    return out['lam_obs'], out['I'], out['sigma_tot']


def degradar_stokes(stokes_dict, lam_grid_nm, instrumento, rng_local=None, fator_ruido=1.0):
    """
    Degrada o vetor de Stokes {I, V, Q, U}.
      'modelo' = 'legado' (INSTRUMENTAL): comportamento antigo, bit a bit
        (inclui o erro de unidade documentado no bloco INSTRUMENTAL);
      'modelo' = 'v2' (INSTRUMENTAL_V2): unidades corretas, grade do instrumento,
        ruído por ponto (constante ou proporcional a I).
    fator_ruido: multiplica o ruído depois do sorteio (1 = real, 0 = sem ruído).
    Retorna dict com 'I','V','Q','U','lam_obs','sigma_tot','alpha'
    (+ 'sigma_por_ponto' no v2).
    """
    if rng_local is None:
        rng_local = np.random.default_rng()
    if _eh_legado(instrumento):
        return _degradar_stokes_legado(stokes_dict, lam_grid_nm, instrumento, rng_local, fator_ruido)
    return _degradar_stokes_v2(stokes_dict, lam_grid_nm, instrumento, rng_local, fator_ruido)


def reamostrar_para_LAM(deg, instrumento, lam_grid=None):
    """
    Leva o perfil observado de volta à grade do modelo (entrada da D²NN).
    Legado: np.interp simples (como generate_degraded_profiles fazia).
    v2: I mantém o valor da borda fora da faixa observada; V vale 0 fora dela
    (não há medida de polarização ali).
    Retorna (I, V) em float64.
    """
    if lam_grid is None:
        lam_grid = LAM
    if _eh_legado(instrumento):
        return (np.interp(lam_grid, deg['lam_obs'], deg['I']),
                np.interp(lam_grid, deg['lam_obs'], deg['V']))
    return (np.interp(lam_grid, deg['lam_obs'], deg['I']),
            np.interp(lam_grid, deg['lam_obs'], deg['V'], left=0.0, right=0.0))

# -----------------------------------------------------------------------
# Modulação Óptica: QWP + Polarizador e Demodulação
# Corresponde à Célula 8 do notebook original.
# -----------------------------------------------------------------------

def get_modulated_intensity(stokes_dict, qwp_angle_deg=45.0, pol_angle_deg=0.0):
    """
    Intensidade detectada após modulação por QWP + polarizador linear.

    Implementa a equação analítica de modulação para um retardador ideal
    de λ/4 seguido de polarizador linear (del Toro Iniesta 2003, Eq. 4.58).

    Parâmetros
    ----------
    stokes_dict   : dict com chaves 'I', 'V' (e opcionalmente 'Q', 'U')
    qwp_angle_deg : ângulo α do eixo rápido da QWP [°]
    pol_angle_deg : ângulo β do eixo de transmissão do polarizador [°]

    Retorna
    -------
    array float32 — intensidade modulada I_det(λ)

    Equação
    -------
    I_det = ½[I + Q·cos2β·cos(2α−2β) + U·sin2β·cos(2α−2β) + V·sin2α·sin(2α−2β)]
    """
    I = stokes_dict['I']
    Q = stokes_dict.get('Q', np.zeros_like(I))
    U = stokes_dict.get('U', np.zeros_like(I))
    V = stokes_dict['V']

    alpha = np.radians(qwp_angle_deg)
    beta  = np.radians(pol_angle_deg)
    sin2a = np.sin(2.0 * alpha)
    cos2b = np.cos(2.0 * beta)
    sin2b = np.sin(2.0 * beta)
    cos2ab = np.cos(2.0 * (alpha - beta))
    sin2ab = np.sin(2.0 * (alpha - beta))

    I_det = 0.5 * (I
                   + Q * cos2b * cos2ab
                   + U * sin2b * cos2ab
                   + V * sin2a * sin2ab)
    return I_det.astype(np.float32)


# --- Asserts de consistência instrumental (rodam sempre que o módulo é importado) ---
_st_circ = {
    'I': np.ones(20, dtype=np.float64),
    'Q': np.zeros(20, dtype=np.float64),
    'U': np.zeros(20, dtype=np.float64),
    'V': np.full(20, 0.1, dtype=np.float64),
}
_lcp = get_modulated_intensity(_st_circ, qwp_angle_deg=45, pol_angle_deg=0)
_rcp = get_modulated_intensity(_st_circ, qwp_angle_deg=45, pol_angle_deg=90)

assert np.allclose(_lcp, 0.5 * (1.0 + 0.1), atol=1e-5), 'Falha: LCP ≠ 0.5(I+V)'
assert np.allclose(_rcp, 0.5 * (1.0 - 0.1), atol=1e-5), 'Falha: RCP ≠ 0.5(I-V)'