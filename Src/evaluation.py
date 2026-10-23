"""
evaluation.py — Avaliação dos modelos e referência clássica.

  - carregar_modelo      : reconstrói D2NN ou D2NNHomodina a partir de um
                           checkpoint e devolve os metadados gravados nele
  - MontadorEntrada      : monta a entrada observável 2D na GPU a partir dos
                           perfis V (equivalente a build_observable_spectrograms)
  - prever, prever_logits: previsões / logits na ordem do DataLoader
  - resumo_classificacao : matriz de confusão, revocação, precisão e F1 por classe
                           e por instrumento; precisao_corrigida para outra
                           proporção entre as classes
  - mcnemar              : comparação pareada de dois classificadores
  - calibracao, ajustar_temperatura, curva_rejeicao : confiança do modelo
  - figura_matrizes_confusao : N matrizes lado a lado
  - amostrar_varredura_bf, limiar_bf50, limiar_bf50_isotonico, curvas_bf : limite de
                           detecção em B·f (a versão isotônica é a validada no script 69)
  - filtro casado        : estimador linear ótimo da amplitude de V (∝ B·f em
                           campo fraco) + limiares — referência clássica para
                           comparar com a D²NN
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.special import expit, logit

from Src.physics import (get_stokes, V_CORE_KMS, INSTRUMENTAL, DLAM, LAM, degradar_stokes,
                         reamostrar_para_LAM)
from Src.optics import SYS4F, FATOR_RAMPA


# -----------------------------------------------------------------------
# Carregamento de modelo a partir do checkpoint
# -----------------------------------------------------------------------
CHAVES_PESADAS = ('model_state_dict', 'optimizer_state_dict', 'historico')


def carregar_modelo(caminho, device, strict=True):
    """
    Reconstrói o modelo gravado em `caminho` e devolve (modelo, metadados).

    A arquitetura vem do campo 'arquitetura' do checkpoint:
      'D2NNHomodina' → Src.model.D2NNHomodina (com campo de referência β)
      'D2NNOptica'   → Src.model.D2NNOptica (máscara de fase + 1 detector; script 62)
      'D2NNOpticaComplexa' → máscara de amplitude e fase + 1 detector (script 74)
      ausente/outro  → Src.model.D2NN (checkpoints anteriores ao script 48)

    metadados : tudo o que foi gravado no checkpoint, exceto pesos, estado do
                otimizador e histórico (ex.: X_scale, entrada, fator_rampa,
                class_defs, beta_ref, epoch, melhor_val_acc).
    strict=True garante que nenhum peso ficou com inicialização aleatória.
    """
    from Src.model import D2NN, D2NNHomodina, D2NNOptica, D2NNOpticaComplexa, PUPIL_T, PHI_VORTICE

    ck = torch.load(caminho, map_location=device, weights_only=False)
    arq = ck.get('arquitetura', 'D2NN')
    if arq == 'D2NNHomodina':
        modelo = D2NNHomodina(pupila=PUPIL_T, phi_vortice=PHI_VORTICE, beta_init=0.0)
    elif arq == 'D2NNOpticaComplexa':
        z = torch.zeros_like(PHI_VORTICE)
        modelo = D2NNOpticaComplexa(z, torch.full_like(z, 0.5), tuple(ck.get('janela', (3, 5))),
                                    treinar_mascara=False, diafragma_pm=ck.get('diafragma_pm'))
    elif arq == 'D2NNOptica':
        modelo = D2NNOptica(torch.zeros_like(PHI_VORTICE), tuple(ck.get('janela', (3, 5))),
                            treinar_fase=False, diafragma_pm=ck.get('diafragma_pm'))
    else:
        modelo = D2NN(pupila=PUPIL_T, phi_vortice=PHI_VORTICE)
    modelo.load_state_dict(ck['model_state_dict'], strict=strict)
    modelo.to(device).eval()

    metadados = {k: v for k, v in ck.items() if k not in CHAVES_PESADAS}
    metadados.setdefault('arquitetura', arq)
    return modelo, metadados


# -----------------------------------------------------------------------
# Entrada observável montada na GPU
# -----------------------------------------------------------------------
class MontadorEntrada:
    """
    X_k(λ) = 0.5 · V(λ) · (a_k − ⟨a⟩) / X_scale,  a_k = linspace(0.8, 1.2, 64)

    Idêntico a Src.optics.build_observable_spectrograms dividido por X_scale
    (o termo de I se cancela na subtração da média sobre as linhas). Permite
    guardar só os perfis V (N × N_LAM) e montar o espectrograma por batch,
    sem ocupar N × 64 × N_LAM na memória da GPU.
    """

    def __init__(self, X_scale, device, n_rows=None, fator_rampa=FATOR_RAMPA):
        if n_rows is None:
            n_rows = SYS4F['det_pixels']
        a = np.linspace(fator_rampa[0], fator_rampa[1], n_rows)
        coef = 0.5 * (a - a.mean()) / X_scale
        self.coef = torch.tensor(coef, dtype=torch.float32, device=device)[None, :, None]

    def __call__(self, V_batch):
        return self.coef * V_batch[:, None, :]      # (B, N_LAM) → (B, n_rows, N_LAM)


# -----------------------------------------------------------------------
# Previsão e resumo
# -----------------------------------------------------------------------
def prever(loader, modelo, transform=None):
    """Previsões na ordem do loader (sem shuffle → ordem dos índices)."""
    modelo.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            X = batch[0] if transform is None else transform(batch[0])
            preds.append(modelo(X).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def resumo_classificacao(y_true, y_pred, inst, nomes, inst_nomes, titulo, verbose=True):
    """
    Matriz de confusão + métricas por classe e por instrumento (dict).

    'por_classe' = revocação (acerto) de cada classe: das amostras da classe,
                   quantas o modelo acertou — não depende da proporção entre
                   as classes no conjunto.
    'precisao'   = das amostras que o modelo chamou de k, quantas são k —
                   DEPENDE da proporção entre as classes; no teste balanceado
                   não vale para um campo solar real (ver precisao_corrigida).
    'f1'         = média harmônica de precisão e revocação.
    Número de classes = len(nomes).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n_cls = len(nomes)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = 100 * (y_true == y_pred).mean()
    erro_padrao = 100 * np.sqrt(acc / 100 * (1 - acc / 100) / len(y_true))
    por_classe = [float(100 * cm[i, i] / max(cm[i].sum(), 1)) for i in range(n_cls)]
    precisao = [float(100 * cm[i, i] / cm[:, i].sum()) if cm[:, i].sum() else float('nan')
                for i in range(n_cls)]
    f1 = [float(2 * p * r / (p + r)) if (p + r) > 0 and np.isfinite(p) else float('nan')
          for p, r in zip(precisao, por_classe)]

    por_inst = {}
    if inst is not None:
        for k, nome in enumerate(inst_nomes):
            m = inst == k
            if not m.any():
                continue
            pc = []
            for c in range(n_cls):
                mc = m & (y_true == c)
                pc.append(float(100 * (y_pred[mc] == c).mean()) if mc.any() else float('nan'))
            por_inst[nome] = {'n': int(m.sum()),
                              'geral': float(100 * (y_true[m] == y_pred[m]).mean()),
                              'por_classe': pc}

    if verbose:
        print(f'\n  {titulo}  —  acurácia geral: {acc:.2f}% (± {erro_padrao:.1f}%, N={len(y_true)})')
        print(f'  {"verdadeiro / previsto":22s}' + ''.join(f'{n[:8]:>10s}' for n in nomes)
              + f'{"acerto":>9s}{"precisão":>10s}{"F1":>8s}')
        for i, n in enumerate(nomes):
            print(f'  {n:22s}' + ''.join(f'{v:>10d}' for v in cm[i])
                  + f'{por_classe[i]:>8.1f}%{precisao[i]:>9.1f}%{f1[i]:>7.1f}%')
        if por_inst:
            print(f'\n  {"Instrumento":12s} {"N":>5s} {"geral":>8s}'
                  + ''.join(f'{n[:8]:>10s}' for n in nomes))
            for nome, d in por_inst.items():
                print(f'  {nome:12s} {d["n"]:>5d} {d["geral"]:>7.1f}%'
                      + ''.join(f'{v:>9.1f}%' for v in d['por_classe']))

    return {'acuracia': float(acc), 'erro_padrao': float(erro_padrao),
            'por_classe': por_classe, 'precisao': precisao, 'f1': f1,
            'f1_macro': float(np.nanmean(f1)),
            'matriz': cm.tolist(), 'por_instrumento': por_inst}


def precisao_corrigida(matriz, prevalencia):
    """
    Precisão de cada classe se a proporção real entre as classes for
    `prevalencia` (soma 1), em vez da proporção do conjunto de teste.

    Usa só as revocações R[j, k] = P(prever k | classe j), que não dependem
    da proporção (regra de Bayes):
        precisão_k = π_k R[k, k] / Σ_j π_j R[j, k]
    """
    cm = np.asarray(matriz, dtype=float)
    R = cm / cm.sum(axis=1, keepdims=True)
    pi = np.asarray(prevalencia, dtype=float)
    pi = pi / pi.sum()
    prever_k = pi @ R
    return [float(100 * pi[k] * R[k, k] / prever_k[k]) if prever_k[k] > 0 else float('nan')
            for k in range(len(pi))]


# -----------------------------------------------------------------------
# Comparação pareada entre dois classificadores (origem: Célula 22C)
# -----------------------------------------------------------------------
def mcnemar(y_true, pred_a, pred_b):
    """
    Teste de McNemar: A e B avaliados nas MESMAS amostras.
      b = A acerta e B erra;  c = A erra e B acerta.
    b + c < 25 → teste binomial exato; senão χ² com correção de continuidade
    (1 grau de liberdade). p < 0,05 → as acurácias diferem de fato.
    """
    from scipy.stats import chi2, binomtest
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    ca, cb = pred_a == y_true, pred_b == y_true
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = b + c
    if n == 0:
        est, p, metodo = 0.0, 1.0, 'sem discordâncias'
    elif n < 25:
        est, p, metodo = float('nan'), float(binomtest(b, n, 0.5).pvalue), 'binomial exato'
    else:
        est = (abs(b - c) - 1) ** 2 / n
        p, metodo = float(chi2.sf(est, 1)), 'χ² com correção de continuidade'
    return {'b_A_acerta_B_erra': b, 'c_A_erra_B_acerta': c,
            'diferenca_pp': float(100 * (ca.mean() - cb.mean())),
            'estatistica': float(est), 'p_valor': p, 'metodo': metodo,
            'significativo_5pct': bool(p < 0.05)}


# -----------------------------------------------------------------------
# Probabilidades, calibração e rejeição (origem: Célula 24, Teste 3)
# -----------------------------------------------------------------------
def prever_logits(loader, modelo, transform=None):
    """Logits na ordem do loader (N × n_classes, numpy)."""
    modelo.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            X = batch[0] if transform is None else transform(batch[0])
            out.append(modelo(X).float().cpu().numpy())
    return np.concatenate(out)


def softmax(logits, T=1.0):
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ajustar_temperatura(logits, y):
    """
    Temperature scaling: o T > 0 que minimiza a log-verossimilhança negativa
    de softmax(logits / T). Ajustar na VALIDAÇÃO, nunca no teste.
    Não muda nenhuma previsão (argmax), só a confiança.
    """
    from scipy.optimize import minimize_scalar
    y = np.asarray(y)

    def nll(logT):
        p = softmax(logits, np.exp(logT))
        return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))

    r = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method='bounded')
    return float(np.exp(r.x))


def calibracao(probs, y, n_bins=10):
    """
    Diagrama de confiabilidade e ECE (erro de calibração esperado).
    Modelo calibrado: das previsões com confiança c, uma fração c está certa.
    ECE = Σ_b (n_b / N) · |acerto_b − confiança_b|.
    """
    probs, y = np.asarray(probs), np.asarray(y)
    conf = probs.max(axis=1)
    certo = probs.argmax(axis=1) == y
    lim = np.linspace(1.0 / probs.shape[1], 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= lim[i]) & ((conf < lim[i + 1]) if i < n_bins - 1 else (conf <= lim[i + 1]))
        if not m.any():
            continue
        cb, ab = float(conf[m].mean()), float(certo[m].mean())
        bins.append({'de': float(lim[i]), 'ate': float(lim[i + 1]), 'n': int(m.sum()),
                     'confianca': cb, 'acerto': ab})
        ece += m.mean() * abs(ab - cb)
    return {'ece': float(ece), 'bins': bins,
            'confianca_media': float(conf.mean()), 'acuracia': float(certo.mean())}


def curva_rejeicao(probs, y, fracoes=None):
    """
    Acurácia das amostras aceitas quando só a fração mais confiante é aceita
    (as demais viram "indeterminado"). Devolve também o limiar de confiança.
    """
    probs, y = np.asarray(probs), np.asarray(y)
    if fracoes is None:
        fracoes = np.linspace(0.1, 1.0, 19)
    conf = probs.max(axis=1)
    certo = probs.argmax(axis=1) == y
    ordem = np.argsort(-conf, kind='stable')
    out = []
    for f in fracoes:
        k = max(1, int(round(f * len(y))))
        sel = ordem[:k]
        out.append({'fracao_aceita': float(k / len(y)), 'acuracia': float(100 * certo[sel].mean()),
                    'limiar_confianca': float(conf[sel].min())})
    return out


# -----------------------------------------------------------------------
# Figura: matrizes de confusão de N classificadores lado a lado (origem: Célula 23)
# -----------------------------------------------------------------------
def figura_matrizes_confusao(resumos, titulos, nomes, caminho, suptitulo='', curtos=None):
    """
    Uma matriz por classificador, cor = revocação (linha normalizada),
    texto = contagem e porcentagem. `resumos` = saídas de resumo_classificacao.
    curtos: rótulos dos eixos (padrão: os próprios nomes).
    """
    import matplotlib.pyplot as plt
    n = len(resumos)
    fig, axs = plt.subplots(1, n, figsize=(5.2 * n, 4.8), constrained_layout=True,
                            squeeze=False)
    if suptitulo:
        fig.suptitle(suptitulo, fontsize=12, fontweight='bold')
    curtos = list(curtos) if curtos is not None else list(nomes)
    for ax, r, t in zip(axs[0], resumos, titulos):
        cm = np.asarray(r['matriz'], dtype=float)
        R = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(R, cmap='Blues', vmin=0, vmax=1)
        for i in range(len(nomes)):
            for j in range(len(nomes)):
                ax.text(j, i, f'{int(cm[i, j])}\n{100 * R[i, j]:.1f}%', ha='center', va='center',
                        fontsize=10, color='white' if R[i, j] > 0.55 else 'black',
                        fontweight='bold' if i == j else 'normal')
        ax.set_xticks(range(len(nomes))); ax.set_xticklabels(curtos)
        ax.set_yticks(range(len(nomes))); ax.set_yticklabels(curtos)
        ax.set_xlabel('previsto'); ax.set_ylabel('verdadeiro')
        ax.set_title(f'{t}\n{r["acuracia"]:.1f}% ± {r["erro_padrao"]:.1f}%  |  F1 macro {r["f1_macro"]:.1f}%',
                     fontsize=10)
    fig.colorbar(im, ax=axs[0].tolist(), shrink=0.8, label='revocação')
    fig.savefig(caminho, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return caminho


# -----------------------------------------------------------------------
# Filtro casado (referência clássica)
# -----------------------------------------------------------------------
def templates_filtro_casado(inst_nomes, v_km_s=V_CORE_KMS, instrumental=None):
    """
    Forma do sinal de Zeeman por unidade de B·f (campo fraco: V ∝ B·f).
    Retorna {índice do instrumento: template (N_LAM,)}.

    instrumental=None (legado, scripts 47–54): V convoluído só com a LSF nominal
      (fwhm_mA·10⁻⁴ nm), sem luz espúria e sem amostragem — por isso o ganho
      g = a/(B·f) medido nesses scripts é < 1 (0,75–0,94).
    instrumental=INSTRUMENTAL_V2: o template é o V esperado SEM ruído, passado pela
      MESMA cadeia dos dados (luz espúria, LSF, amostragem e volta à grade LAM);
      então a = B·f exatamente no campo fraco (g = 1).
    """
    if instrumental is not None:
        st = get_stokes(100.0, filling=1.0, v_km_s=v_km_s)
        rng = np.random.default_rng(0)
        out = {}
        for k, n in enumerate(inst_nomes):
            deg = degradar_stokes(st, LAM, instrumental[n], rng_local=rng, fator_ruido=0.0)
            out[k] = reamostrar_para_LAM(deg, instrumental[n])[1] / 100.0
        return out
    t0 = get_stokes(100.0, filling=1.0, v_km_s=v_km_s)['V'] / 100.0
    return {k: gaussian_filter1d(t0, INSTRUMENTAL[n]['fwhm_mA'] * 1e-4 / 2.3548 / DLAM)
            for k, n in enumerate(inst_nomes)}


def amplitude_filtro_casado(V, inst_idx, templates, modulo=False):
    """
    Projeção de V sobre o template do instrumento → estimativa de B·f.

    modulo=False (padrão, como nos scripts 47–49): amplitude COM sinal —
        uma região de polaridade negativa dá amplitude negativa e cai em
        "Quieta". O dataset de treino só tem polaridade positiva.
    modulo=True : |amplitude| — invariante à polaridade, como a física exige
        (a classe depende de |B|·f). Os limiares precisam ser ajustados na
        mesma convenção.
    """
    out = np.empty(len(V))
    for i in range(len(V)):
        t = templates[int(inst_idx[i])]
        out[i] = V[i] @ t / (t @ t)
    return np.abs(out) if modulo else out


def _limiar(v0, v1):
    """Limiar que maximiza o acerto entre duas classes ordenadas (v0 < v1)."""
    c = np.sort(np.concatenate([v0, v1]))
    acertos = [(v0 < t).sum() + (v1 >= t).sum() for t in c]
    return float(c[int(np.argmax(acertos))])


def ajustar_limiares(a, y, inst=None):
    """
    Limiares Quieta/Rede e Rede/Mancha sobre a amplitude estimada.
    inst=None → um par global (como a D²NN, que não sabe o instrumento);
    caso contrário, um par por instrumento.
    """
    if inst is None:
        return (_limiar(a[y == 0], a[y == 1]), _limiar(a[y == 1], a[y == 2]))
    return {int(k): ajustar_limiares(a[inst == k], y[inst == k]) for k in np.unique(inst)}


def classificar_filtro_casado(a, limiares, inst=None):
    """Aplica os limiares (globais ou por instrumento) às amplitudes."""
    if inst is None:
        t1, t2 = limiares
        return np.where(a < t1, 0, np.where(a < t2, 1, 2))
    pred = np.empty(len(a), dtype=int)
    for k, (t1, t2) in limiares.items():
        m = inst == k
        pred[m] = np.where(a[m] < t1, 0, np.where(a[m] < t2, 1, 2))
    return pred


# -----------------------------------------------------------------------
# Limite de detecção em B·f (origem: scripts 53–54; substitui Células 28–30)
# -----------------------------------------------------------------------
FAIXAS_BF_CLASSES = {0: (0.05, 5.0), 1: (5.0, 150.0), 2: (150.0, 2500.0)}   # G


def amostrar_varredura_bf(n_por_classe, seed, classes=(0, 1, 2), faixas=None):
    """
    (B, f, classe) com B·f log-uniforme em cada classe e (B, f) sorteado DENTRO
    da caixa da classe (CLASS_DEFS) — rótulo correto por construção.
    faixas: {classe: (B·f mín, B·f máx)} em G; padrão FAIXAS_BF_CLASSES.
    Com classes=(0, 1, 2) reproduz exatamente a varredura do script 53.
    """
    from Src.dataset import CLASS_DEFS
    faixas = faixas or FAIXAS_BF_CLASSES
    rng = np.random.default_rng(seed)
    B, F, y = [], [], []
    for b_min, b_max, f_min, f_max, cls, _ in CLASS_DEFS:
        if cls not in classes:
            continue
        lo, hi = faixas[cls]
        bf = np.exp(rng.uniform(np.log(lo), np.log(hi), n_por_classe))
        f_lo = np.maximum.reduce([np.full(n_por_classe, f_min), bf / b_max,
                                  np.full(n_por_classe, 1e-4)])
        f_hi = np.minimum(f_max, bf / b_min) if b_min > 0 else np.full(n_por_classe, f_max)
        f_hi = np.maximum(f_hi, f_lo)
        f = rng.uniform(f_lo, f_hi)
        B.append(bf / f); F.append(f); y.append(np.full(n_por_classe, cls))
    B, F, y = map(np.concatenate, (B, F, y))
    for b_min, b_max, f_min, f_max, cls, _ in CLASS_DEFS:        # dentro das caixas
        m = y == cls
        if not m.any():
            continue
        assert (B[m] >= b_min * (1 - 1e-9)).all() and (B[m] <= b_max * (1 + 1e-9)).all()
        assert (F[m] >= f_min - 1e-12).all() and (F[m] <= f_max + 1e-12).all()
    return B, F, y


def _ajuste_psicometrico(x, y, piso):
    """
    Máxima verossimilhança de P(x) = γ + (1 − γ)·σ(b·(x − x₀)), b > 0.
    piso=None → γ ajustado em [0, 0,5); piso=float → γ fixo.
    Devolve log10(B·f) em que P = 50% (ou NaN).
    """
    livre = piso is None

    def prob(p):
        g = 0.5 * expit(p[2]) if livre else piso
        return g + (1.0 - g) * expit(np.exp(p[1]) * (x - p[0])), g

    def nll(p):
        P = np.clip(prob(p)[0], 1e-9, 1.0 - 1e-9)
        return -np.sum(y * np.log(P) + (1.0 - y) * np.log1p(-P))

    iniciais = [[x0, lb] + ([c] if livre else [])
                for x0 in np.quantile(x, np.linspace(0.05, 0.95, 13))
                for lb in np.log([1.0, 3.0, 10.0])
                for c in ((-3.0, 0.0) if livre else (None,))]
    p = minimize(nll, min(iniciais, key=nll), method='Nelder-Mead',
                 options={'xatol': 1e-4, 'fatol': 1e-7, 'maxiter': 2000}).x
    g = prob(p)[1]
    if g >= 0.5:
        return np.nan
    return p[0] + logit((0.5 - g) / (1.0 - g)) / np.exp(p[1])


def limiar_bf50(bf, y01, rng, n_boot=200, piso=None):
    """
    B·f em que P(y01 = 1) = 50%, com IC 95% por bootstrap.

    Modelo (função psicométrica): P = γ + (1 − γ)·σ(b·(log10 B·f − x₀)).
    γ é o piso de falsos positivos, P quando B·f → 0:
      piso=None  → γ ajustado junto (0 ≤ γ < 0,5): detecção por argmax (script 53);
      piso=float → γ fixo: detecção com falso alarme fixado (script 54, γ = FAP).

    A versão anterior usava a logística sem piso (P → 0 quando B·f → 0). Com
    5–35% de falsos positivos no piso, o ajuste ficava raso e deslocava o 50%.
    No filtro casado, cujo B·f₅₀ é conhecido analiticamente (1,96·σ_a/ganho a
    5% de falso alarme, linear no ruído), o erro ia de −35% no meio da faixa a
    +55% perto da borda. Validação no script 54 (coluna "teoria").

    Fora da faixa amostrada devolve NaN com o motivo (o ajuste extrapolaria);
    o teste usa as pontas empíricas (10% menores e maiores B·f).
    Retorna (estimativa, IC inferior, IC superior, motivo).
    """
    bf = np.asarray(bf, float)
    y = np.asarray(y01, float)
    x = np.log10(bf)
    nan3 = (float('nan'),) * 3
    if y.min() == y.max():
        return nan3 + ('sem transição (uma só resposta)',)
    ordem = np.argsort(x)
    n10 = max(10, len(x) // 10)
    if y[ordem[:n10]].mean() >= 0.5:
        return nan3 + ('abaixo da faixa (P > 50% já no menor B·f)',)
    if y[ordem[-n10:]].mean() < 0.5:
        return nan3 + ('acima da faixa (P < 50% até o maior B·f)',)
    x50 = _ajuste_psicometrico(x, y, piso)
    if not np.isfinite(x50):
        return nan3 + ('sem transição (piso ≥ 50%)',)
    est = 10 ** x50
    if not (bf.min() <= est <= bf.max()):
        motivo = ('abaixo da faixa (P > 50% já no menor B·f)' if est < bf.min()
                  else 'acima da faixa (P < 50% até o maior B·f)')
        return nan3 + (motivo,)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if y[i].min() != y[i].max():
            boots.append(_ajuste_psicometrico(x[i], y[i], piso))
    boots = np.asarray(boots, float)
    boots = boots[np.isfinite(boots)]
    if len(boots) < max(2, n_boot // 2):
        return float(est), float('nan'), float('nan'), ''
    lo, hi = 10 ** np.percentile(boots, [2.5, 97.5])
    return float(est), float(lo), float(hi), ''


def _x50_isotonico(x, y):
    """
    log10(B·f) em que a curva de detecção cruza 50%, por regressão isotônica (PAV,
    crescente): blocos de valor constante → centros (média de x no bloco) → interpolação
    linear do nível 0,5 entre os centros. Sem forma funcional. NaN se não cruza.
    """
    from sklearn.isotonic import IsotonicRegression
    ordem = np.argsort(x)
    xs, ys = x[ordem], y[ordem]
    f = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True).fit_transform(xs, ys)
    quebra = np.flatnonzero(np.diff(f) != 0) + 1
    blocos = np.split(np.arange(len(xs)), quebra)
    niveis = np.array([f[b[0]] for b in blocos])
    centros = np.array([xs[b].mean() for b in blocos])
    if niveis[0] >= 0.5 or niveis[-1] < 0.5:
        return np.nan
    return float(np.interp(0.5, niveis, centros))


def limiar_bf50_isotonico(bf, y01, rng, n_boot=200):
    """
    B·f em que P(detecção) = 50%, por regressão isotônica em log10(B·f), com IC 95% por
    bootstrap. Mesma interface e mesmas regras de "fora da faixa" que limiar_bf50.
    Motivo (script 69): a forma logística em log(B·f) de limiar_bf50 não é a forma real da
    curva; com a varredura de densidade diferente abaixo e acima de 5 G e cortada em 150 G,
    o ajuste subestimava B·f₅₀ em 10–20% acima de ~15 G.
    Retorna (estimativa, IC inferior, IC superior, motivo).
    """
    bf = np.asarray(bf, float)
    y = np.asarray(y01, float)
    x = np.log10(bf)
    nan3 = (float('nan'),) * 3
    if y.min() == y.max():
        return nan3 + ('sem transição (uma só resposta)',)
    ordem = np.argsort(x)
    n10 = max(10, len(x) // 10)
    if y[ordem[:n10]].mean() >= 0.5:
        return nan3 + ('abaixo da faixa (P > 50% já no menor B·f)',)
    if y[ordem[-n10:]].mean() < 0.5:
        return nan3 + ('acima da faixa (P < 50% até o maior B·f)',)
    x50 = _x50_isotonico(x, y)
    if not np.isfinite(x50):
        return nan3 + ('sem transição',)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if y[i].min() != y[i].max():
            boots.append(_x50_isotonico(x[i], y[i]))
    boots = np.asarray(boots, float)
    boots = boots[np.isfinite(boots)]
    if len(boots) < max(2, n_boot // 2):
        return float(10 ** x50), float('nan'), float('nan'), ''
    lo, hi = 10 ** np.percentile(boots, [2.5, 97.5])
    return float(10 ** x50), float(lo), float(hi), ''


def _ajuste_fisico(x, y, z_c, beta_livre):
    """
    Máxima verossimilhança de P(x) = Φ(u − z_c) + Φ(−u − z_c), u = (x/s)^β (x = B·f linear).
    É a curva exata de um detector |estatística linear| com ruído gaussiano e limiar fixado
    pelo falso alarme (z_c = Φ⁻¹(1 − FAP/2)): β = 1. β livre acomoda detectores quadráticos.
    Devolve (log s, log β) — log β = 0 se beta_livre=False — e a NLL mínima.
    """
    from scipy.stats import norm
    lx = np.log(np.maximum(x, 1e-300))

    def prob(p):
        u = np.exp(np.clip((np.exp(p[1]) if beta_livre else 1.0) * (lx - p[0]), -50.0, 50.0))
        return norm.sf(z_c - u) + norm.cdf(-z_c - u)

    def nll(p):
        P = np.clip(prob(p), 1e-9, 1.0 - 1e-9)
        return -np.sum(y * np.log(P) + (1.0 - y) * np.log1p(-P))

    iniciais = [[ls, lb] for ls in np.log(np.quantile(x, np.linspace(0.05, 0.95, 9)))
                for lb in ((np.log(0.7), 0.0, np.log(1.4)) if beta_livre else (0.0,))]
    if beta_livre:
        r = minimize(nll, min(iniciais, key=nll), method='Nelder-Mead',
                     options={'xatol': 1e-5, 'fatol': 1e-8, 'maxiter': 3000})
        return r.x[0], r.x[1], float(r.fun)
    f1 = lambda q: nll([q[0], 0.0])
    r = minimize(f1, [min(iniciais, key=nll)[0]], method='Nelder-Mead',
                 options={'xatol': 1e-6, 'fatol': 1e-9, 'maxiter': 2000})
    return r.x[0], 0.0, float(r.fun)


def limiar_bf50_fisico(bf, y01, rng, n_boot=200, fap=0.05, beta_livre=False, info=None):
    """
    B·f em que P(detecção) = 50%, pela forma física da curva de detecção, com IC 95% por
    bootstrap. Mesma interface e mesmas regras de "fora da faixa" que limiar_bf50.

    Modelo: P = Φ(u − z_c) + Φ(−u − z_c), u = (B·f/s)^β, z_c = Φ⁻¹(1 − FAP/2).
    O piso P(0) = FAP é imposto pela forma (não é ajustado). β = 1 (padrão): um só
    parâmetro livre, s; B·f₅₀ = s·z₅₀ com z₅₀ = 1,960 a FAP = 5%.
    Motivo (scripts 69 e 70): a logística em log B·f subestimava B·f₅₀ em ~11% acima de
    15 G (forma errada) e a isotônica tinha dispersão de ~9% (sem forma).
    info : dict opcional; recebe 'beta' e 'nll' do ajuste central.
    Retorna (estimativa, IC inferior, IC superior, motivo).
    """
    from scipy.optimize import brentq
    from scipy.stats import norm
    z_c = norm.isf(fap / 2)
    z50 = brentq(lambda m: norm.sf(z_c - m) + norm.cdf(-z_c - m) - 0.5, 0.0, 10.0)
    bf = np.asarray(bf, float)
    y = np.asarray(y01, float)
    nan3 = (float('nan'),) * 3
    if y.min() == y.max():
        return nan3 + ('sem transição (uma só resposta)',)
    ordem = np.argsort(bf)
    n10 = max(10, len(bf) // 10)
    if y[ordem[:n10]].mean() >= 0.5:
        return nan3 + ('abaixo da faixa (P > 50% já no menor B·f)',)
    if y[ordem[-n10:]].mean() < 0.5:
        return nan3 + ('acima da faixa (P < 50% até o maior B·f)',)

    def x50(ls, lb):
        return float(np.exp(ls) * z50 ** (1.0 / np.exp(lb)))

    ls, lb, f = _ajuste_fisico(bf, y, z_c, beta_livre)
    est = x50(ls, lb)
    if info is not None:
        info.update({'beta': float(np.exp(lb)), 'nll': f})
    if not (bf.min() <= est <= bf.max()):
        motivo = ('abaixo da faixa (P > 50% já no menor B·f)' if est < bf.min()
                  else 'acima da faixa (P < 50% até o maior B·f)')
        return nan3 + (motivo,)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(bf), len(bf))
        if y[i].min() != y[i].max():
            boots.append(x50(*_ajuste_fisico(bf[i], y[i], z_c, beta_livre)[:2]))
    boots = np.asarray(boots, float)
    boots = boots[np.isfinite(boots)]
    if len(boots) < max(2, n_boot // 2):
        return est, float('nan'), float('nan'), ''
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi), ''


def curvas_bf(bf, pred, bins):
    """Fração de cada classe prevista por intervalo de B·f."""
    idx = np.digitize(bf, bins) - 1
    out = []
    for i in range(len(bins) - 1):
        m = idx == i
        n = int(m.sum())
        frac = [float((pred[m] == c).mean()) if n else float('nan') for c in range(3)]
        out.append({'bf_centro': float(np.sqrt(bins[i] * bins[i + 1])), 'n': n, 'frac': frac})
    return out
