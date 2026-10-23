# D²NN puramente óptica: ablação, máscara de fase, diafragma, Doppler, inclinação, máscara de amplitude e fase e referências do artigo B (scripts 61–76)

**Projeto:** PHOTON ROAD — D²NN para classificação de perfis Stokes Ca II 854,2 nm
**Escopo:**
- quanto da classificação vinha da parte óptica dos modelos híbridos (script 61);
- uma D²NN em que a óptica faz todo o trabalho: uma máscara de fase, um detector e dois
  limiares (script 62);
- o limite de detecção dessa D²NN e o diafragma de campo que corrige o ViSP (54, 67);
- a robustez ao deslocamento Doppler (script 68);
- o estimador de B·f₅₀ usado nos números finais (scripts 69 e 70; detalhes em
  `ROBUSTEZ_E_DETECCAO.md`, seção 6.6);
- a inclinação do campo θ (script 71);
- a origem da perda no ViSP e a D²NN com máscara de amplitude e fase (scripts 73 e 74);
- o filtro casado branqueado como referência do ótimo, a fração de sinal e de ruído transmitida
  pelas máscaras e a quantização de fase do SLM (script 75);
- a máscara de amplitude e fase projetada como filtro branqueado (script 76).

**Relatórios:** `Outputs/Reports/61_ablacao_optica.json`, `62_d2nn_optica_pura.json`,
`54_snr_detection_v2_62.json`, `67_d2nn_optica_diafragma.json`, `54_snr_detection_v2_67.json`,
`68_doppler.json`, `69_validacao_estimador.json`, `70_estimador_fisico.json`,
`71_inclinacao.json`, `73_mascara_visp.json`, `74_d2nn_complexa.json`, `54_snr_detection_v2_74.json`,
`75_referencias_artigo_b.json`, `76_mascara_branqueada.json`
**Relacionados:** `docs/MODELO_INSTRUMENTAL.md` (instrumento v2 e scripts 57–60),
`docs/ROBUSTEZ_E_DETECCAO.md` (método de detecção a falso alarme fixo),
`docs/RETREINO_ENTRADA_OBSERVAVEL.md` (modelos 45–52, achados 5 e 6)
**Modelos de referência v2:** `CheckPoints/modelo_d2nn_optico_v2.pt` (script 67: só fase, diafragma
±200 pm) e `CheckPoints/modelo_d2nn_complexo_v2.pt` (script 74: amplitude e fase)

---

## Resumo

1. **Nos modelos híbridos a óptica não fazia nada** (script 61). Zerar a fase aprendida ou
   trocar o vórtice por uma pupila simples muda a acurácia em ≤ 0,43 pp (p ≥ 0,17) no 47, no
   57 e no 59. A correção de fase aprendida tem desvio padrão de 2 a 10 mrad. O sistema
   efetivo era imageamento 4f fixo + uma cabeça eletrônica de ~4,2 M parâmetros.
2. **Uma D²NN puramente óptica empata com o filtro casado** (script 62). Uma máscara de fase,
   a energia numa janela de 3 × 5 pixels e dois limiares (6 parâmetros eletrônicos) dão
   86,67% no teste grande v2, contra 86,13% do filtro |a| (p = 0,25) e 76,40% do híbrido 57
   (+10,27 pp, p = 1,8×10⁻³⁵). É exatamente invariante à polaridade.
3. **No critério de detecção (5% de falso alarme), com o diafragma de campo** (scripts 67, 54
   e 70), a razão B·f₅₀ D²NN / filtro |a| tem mediana 1,05 no IBIS (1,02–1,15 em k),
   **0,82 no CRISP** (0,77–0,91) e 1,38 no ViSP (1,20–1,51). Em k = 1, com IC 95% pareado:
   1,05 [0,95; 1,15], **0,87 [0,81; 0,94]** e 1,20 [1,02; 1,50]. O híbrido 57 ficava em 1,4–2,0.
4. **No CRISP a D²NN óptica supera o filtro |a|**, porque aprende a ponderação do ruído
   proporcional a I que o |a| simples não faz. Contra o filtro casado branqueado (o ótimo, script
   75: 47,5 G em k = 1), a razão é 1,06 [0,98; 1,14] em k = 1 e 1,05 de mediana em k: a máscara
   recupera 74% do ganho do branqueamento e fica a ~5% do ótimo. (A versão anterior citava
   49,3 G como ótimo ponderado; o valor de referência passa a ser o do script 75.)
5. **A máscara de fase pura não atenua o ruído longe da linha**, e é isso que limitava o ViSP.
   Um diafragma de campo (fenda no plano de entrada) resolve a maior parte.
6. **Doppler não é problema** (script 68): até ±3 km/s o B·f₅₀ da D²NN muda ≤ 2%; em 5 km/s,
   +5 a +11%, como o filtro de template fixo. Uma fileira de detectores não é necessária.
7. **Os B·f₅₀ vêm de um estimador validado** (scripts 69 e 70): a forma exata da curva de
   detecção de um detector linear, que também descreve a D²NN óptica (β = 1,03).
8. **Os limites valem para B_los·f** (script 71): razão 1,000 de θ = 0° a 75° para a D²NN e o
   filtro.
9. **A perda no ViSP é da máscara só de fase** (script 73): o POF não passa de 1,56× o filtro,
   mesmo calculado para o ViSP; a máscara casada de amplitude e fase (H = S*) reproduz o filtro
   (0,99–1,00×).
10. **Com amplitude e fase, uma única máscara atinge o filtro casado nos três instrumentos**
    (scripts 74 e 54): razão mediana 1,01 (IBIS), 0,89 (CRISP) e 1,00 (ViSP), contra 1,05 /
    0,82 / 1,38 da máscara só de fase. Troca: a só de fase ganha mais no CRISP; a complexa
    resolve o ViSP. Transmissão da máscara: 3%.
11. **Contra o ótimo branqueado** (script 75), a máscara complexa atinge o ótimo onde o ruído é
    branco (IBIS 1,03, ViSP 1,00) mas não no CRISP (1,15 [1,08; 1,26] em k = 1); a só de fase
    faz o contrário. Nenhuma das duas, *treinada*, faz as duas coisas (ver item 14).
12. **A máscara de amplitude é seletiva** (script 75): transmite 24–44% da energia do sinal e só
    1,6–9,6% da do ruído, embora a transmissão média seja de 3%.
13. **A quantização de fase não é limitante** (script 75): 3 bits bastam para o 67 e 2 bits para
    o 74 (perda de B·f₅₀ ≤ 5%), sem retreino.
14. **Uma máscara de amplitude e fase projetada realiza o ótimo** (script 76). O filtro casado
    branqueado feito opticamente (H = conj F{X_w}, w o peso branqueado exato) dá 1,00 / 1,00 /
    1,00 do ótimo (IBIS / CRISP / ViSP, mediana em k). A falta de branqueamento do 74 é do treino,
    que parte de H = S*, e não da arquitetura. Uma máscara única para os três instrumentos custa
    3 / 11 / 3%; como o SLM é reprogramável, a máscara por instrumento não tem custo.

---

## 1. Ablação óptica dos modelos híbridos (script 61)

Substitui as Células 36, 36B, 36_FIX e 37 do Colab. A cabeça eletrônica fica fixa e só a
óptica muda. Teste grande de cada modelo (v2 para o 57 e o 59, legado para o 47).

| Variante | 47 | 57 | 59 |
|---|---|---|---|
| V0 treinado | 75,23% | 76,40% | 90,07% |
| V1 vórtice estático (Δφ = 0) | −0,07 pp (p = 0,77) | −0,10 pp (p = 0,25) | −0,43 pp (p = 0,17) |
| V2 só pupila (fase total = 0) | −0,13 pp (p = 0,74) | −0,10 pp (p = 0,38) | −0,43 pp (p = 0,17) |
| V3 fase aleatória | −41,4 pp | −6,2 pp | −57,5 pp |
| V4 Δφ × 10 | −0,13 pp | +0,13 pp | −14,1 pp |
| std(Δφ) por camada | 2,7 mrad | 2,2 mrad | 9,6 mrad |

- **V1 ≈ V2 ≈ V0:** nem a fase aprendida nem o vórtice contribuem.
- **V3** não mede o vórtice: uma fase aleatória é um difusor que embaralha o espectrograma, e a
  cabeça foi treinada com outra óptica.
- **V4 no 59:** amplificar a correção muda a fase da referência homodina (frequência zero).
- **As três camadas são idênticas por construção.** Em série no mesmo plano de Fourier, a perda
  só depende da fase total; as três recebem o mesmo gradiente e, partindo de zero, seguem
  iguais (diferenças de ~10⁻⁷ rad). Confirma o achado 5.
- **Por que a óptica não aprende:** a cabeça eletrônica compensa qualquer óptica, e não sobra
  gradiente consistente para as fases.

## 2. Física da leitura óptica

- **As classes têm a mesma forma de V** (achado 18): V ∝ B·f·dI/dλ no campo fraco. Com óptica
  linear e detecção |E|², o padrão de intensidade no detector só muda de escala (c²) entre as
  classes. Razões de energia entre regiões (leitura do Lin et al. 2018) não carregam
  informação; toda leitura óptica se reduz a **um escalar**.
- **O ótimo é concentrar a projeção casada num ponto**, onde |E|² = |a|². Decidir por |a|² é o
  mesmo que decidir por |a| (achado 6 revisado): a detecção de intensidade não perde nada em
  princípio.
- **Uma máscara de fase pura faz isso:** o filtro casado de fase (POF), H = S*/|S|, leva a saída
  a IFFT(|S|), um pico na origem. O ruído branco continua branco (a máscara tem ganho 1).
- **A eficiência do POF é menor que 1** em relação ao filtro casado de amplitude, porque ele não
  pondera as frequências por |S|. O treino da fase recupera parte disso.

## 3. D²NN puramente óptica (script 62)

**Modelo (`Src.model.D2NNOptica`):** X (64 × N_LAM) → uma `CamadaFase` no plano de Fourier,
dentro da pupila → |E|² → energia E numa janela de 3 × 5 pixels centrada no pico →
logits_k = w_k·z + b_k, com z = (log E − μ)/s. A "eletrônica" são 6 números: dois limiares
ordinais sobre um único detector, ou seja, um comparador.

**Treino:** protocolo do 57 (mesmo dataset, split e ruído renovado a cada época), Adam com
lr = 10⁻² na fase e na leitura, parada após 15 épocas sem melhora (`Src.treino_optico`).

| Método (teste grande v2) | Geral | Quieta | Rede | Mancha | IBIS | CRISP | ViSP |
|---|---|---|---|---|---|---|---|
| D²NN óptica, POF treinado | **86,67%** | 81,6 | **79,7** | 98,7 | 87,1 | 79,5 | 95,4 |
| D²NN óptica, POF fixo (analítico) | 83,10% | 76,3 | 74,0 | 99,0 | 81,3 | 77,8 | 92,7 |
| Filtro casado \|a\| | 86,13% | 83,8 | 76,0 | 98,6 | 88,5 | 78,6 | 92,3 |
| Filtro casado com sinal | 89,13% | 87,3 | 81,5 | 98,6 | 90,2 | 83,0 | 95,4 |
| D²NN 57 (híbrida) | 76,40% | 94,1 | 36,9 | 98,2 | 78,1 | 74,4 | 76,4 |

- POF treinado × filtro |a|: +0,53 pp (p = 0,25); × 57: +10,27 pp (p = 1,8×10⁻³⁵).
- POF fixo × 57: +6,70 pp (p = 1,4×10⁻¹⁵) — **sem nenhum treino da óptica**.
- Polaridade: +V e −V idênticos (100% das previsões inalteradas) nas duas variantes.
- A fase treinada se afasta do POF de verdade: std(Δφ) = 1,18 rad, contra 2–10 mrad nos híbridos.
- Critérios definidos antes (C1 ≥ |a| − 3 pp e Mancha ≥ 95%; C2 > 57; C3 invariância):
  todos atendidos.

## 4. Limite de detecção e o diafragma de campo (54 e 67)

### 4.1 Sem diafragma (54 com o 62)

Razão B·f₅₀(D²NN) / B·f₅₀(filtro |a|) a 5% de falso alarme, k = 0,1–1,4:

| | IBIS | CRISP | ViSP |
|---|---|---|---|
| D²NN 62 (óptica) | 1,00–1,09 | 0,84–0,97 | 1,5–2,0 |
| D²NN 57 (híbrida), para comparação | 1,5–1,7 | 1,4–1,5 | 1,6–2,0 |

**Por que o ViSP fica para trás:** a máscara de fase tem ganho 1 em todas as frequências e não
atenua o ruído longe da linha; o filtro casado atenua (peso zero onde o template é zero). No
IBIS e no CRISP o V é zero fora da faixa medida (não há ruído ali). No ViSP o ruído cobre a
janela inteira de ±5 Å e chega ao detector.

### 4.2 Diafragma de campo (script 67)

Uma fenda no plano de entrada que só deixa passar |λ − λ_núcleo| ≤ D. O POF é recalculado com
o template multiplicado pela mesma janela; a fase é treinada.

| Variante | Acurácia | IBIS | CRISP | ViSP | Razão média |
|---|---|---|---|---|---|
| 62 (sem diafragma) | 86,67% | 1,01 | 0,88 | 1,51 | 1,13 |
| **±200 pm (escolhida)** | 86,03% | 1,04 | 0,84 | 1,20 | 1,03 |
| ±120 pm | 85,60% | 1,03 | 0,85 | 1,10 | 0,99 |
| ±80 pm | 84,97% | 1,09 | 0,82 | 1,22 | 1,04 |

(k = 1; B·f₅₀ do filtro |a|: 26,8 / 51,5 / 8,1 G.)

- **Critérios (definidos antes):** C1, ViSP ≤ 1,2: 1,20, atendido no limite; C2, IBIS e CRISP sem
  piora: atendido.
- **Regra de escolha (definida antes):** menor razão média com acurácia ≥ 62 − 1 pp. O ±120 pm
  tem a menor razão média (0,99), mas ficou fora por 0,07 pp de acurácia (85,60% contra o mínimo
  de 85,67%), diferença bem dentro do erro padrão de ~0,6 pp. Mantida a escolha pré-definida;
  o ±120 pm fica como análise de sensibilidade.
- **±80 pm corta parte do lobo de V** e volta a piorar: existe uma largura ótima.

### 4.3 Curva completa do modelo escolhido (54 com o 67)

Estimador físico (β = 1, script 70; `ROBUSTEZ_E_DETECCAO.md`, seção 6.6). Razões de k = 0,1 a 2
com IC 95% por bootstrap pareado (mesmos perfis reamostrados para os dois detectores):

| Instrumento | Razão 67 / filtro \|a\|: mediana (faixa em k) | Razão em k = 1 [IC 95%] | B·f₅₀ em k = 1 (67 / filtro / teoria) |
|---|---|---|---|
| IBIS | 1,05 (1,02–1,15) | 1,05 [0,95; 1,15] | 31,1 / 29,6 / 31,3 G |
| CRISP | **0,82 (0,77–0,91)** | **0,87 [0,81; 0,94]** | 50,2 / 57,8 / 57,8 G |
| ViSP | 1,38 (1,20–1,51) | 1,20 [1,02; 1,50] | 10,3 / 8,6 / 7,5 G |

- **IBIS:** equivalente ao filtro (o IC em k = 1 contém 1).
- **CRISP:** melhor que o filtro em todos os k (IC abaixo de 1). O ótimo ponderado (peso 1/σ_k²,
  pontos nativos) era citado como 49,3 G (`MODELO_INSTRUMENTAL.md`, seção 5). Com o filtro
  branqueado calculado nos mesmos perfis e com o mesmo estimador (script 75, seção 4.9), o
  ótimo é 47,5 G, e a D²NN, com 50,2 G, fica a 6% dele (IC de 67/ótimo contém 1).
- **ViSP:** pior que o filtro em todos os k (IC acima de 1). A razão não é constante (1,3–1,5
  com pouco ruído, 1,2 perto do ruído real): uma perda que não escala com σ. A hipótese é a
  máscara única para os três instrumentos (o ViSP tem a LSF mais estreita).
- **Planejamento do CRISP (grade provisória):** com a média de 11 varreduras, B·f₅₀ = 14,8 G
  (filtro: 19,2 G; teoria do filtro: 17,7 G); com 11 varreduras + 2×2, 13,7 G (filtro 16,9 G).
  Rede de 20–40 G detectada em 95–96%.
- **Fora do ruído de treino** (k = 1,4–2), a acurácia por argmax cai mais que a do filtro: o
  ponto de operação acompanha o ruído do treino (pendência: k sorteado).
- **Mudança em relação à versão anterior deste documento** (estimador psicométrico): as razões
  medianas não mudaram (1,05 / 0,82 / 1,38); os B·f₅₀ absolutos acima de ~15 G subiram ~10%
  (ex.: IBIS em k = 1, D²NN de 28,0 para 31,1 G; filtro de 26,8 para 29,6 G).

### 4.4 Doppler (script 68)

O 4f é equivariante a translações em λ: um desvio Doppler desloca o pico do POF (8,55 px por
km/s). Com a janela fixa, a energia lida é a correlação com o template no atraso zero, que é
exatamente o que faz o filtro de template fixo. Os dois perdem sinal na mesma proporção, e a
perda é pequena porque o núcleo de Ca II 854,2 nm é largo comparado com alguns km/s.

Perfis v2 com ruído real (k = 1); B·f₅₀ relativo a Δv = 0 do próprio método (estimador
psicométrico — os valores relativos não dependem do estimador):

| Δv [km/s] | IBIS: 67 / fileira / filtro / banco | CRISP: 67 / fileira / filtro / banco | ViSP: 67 / fileira / filtro / banco |
|---|---|---|---|
| +1 | 1,00 / 1,00 / 1,00 / 1,02 | 1,02 / 1,02 / 1,01 / 0,98 | 1,00 / 1,01 / 1,00 / 0,98 |
| +3 | 1,02 / 1,00 / 1,01 / 1,01 | 1,00 / 1,07 / 1,02 / 0,97 | 1,00 / 1,00 / 1,00 / 1,00 |
| +5 | 1,05 / 1,04 / 1,02 / 1,06 | 1,11 / 1,10 / 1,08 / 1,02 | 1,00 / 1,04 / 1,00 / 1,02 |
| −2 | 1,02 / 1,03 / 1,01 / 1,02 | 1,02 / 0,98 / 0,99 / 0,99 | 1,00 / 1,02 / 1,00 / 1,01 |

(fileira: 19 detectores de −45 a +45 px, escore = máximo; banco: templates de −6 a +6 km/s,
escore = máximo |a|.)

- **Acurácia (argmax):** 86,3% em Δv = 0, 85,9% em 3 km/s, 84,9% em 5 km/s (filtro: 85,7 →
  85,5 → 85,0%).
- **Critério CD (definido antes):** fileira com B·f₅₀ ≤ 1,1 × o do banco para |Δv| ≤ 3 km/s no
  IBIS e no CRISP — **não atendido** (6 de 12 casos). O critério estava mal posto: compara
  valores absolutos e por isso mistura a eficiência de base (no IBIS, em Δv = 0, a fileira já
  está 1,18× acima do banco, pelo limiar mais alto do máximo de 19 detectores) com a
  robustez ao Doppler, que era a pergunta. Fica registrado como não atendido.
- **Conclusão:** na faixa de velocidades do núcleo da cromosfera calma e da Rede, o detector
  único basta. A fileira só pagaria um limiar mais alto.

### 4.5 Estimador de B·f₅₀ (scripts 69 e 70)

Resumo (detalhes em `ROBUSTEZ_E_DETECCAO.md`, seção 6.6): a logística em log B·f subestimava
B·f₅₀ em ~11% acima de 15 G; a regressão isotônica tinha dispersão de ~9%. O estimador
adotado usa a forma exata da curva de detecção de um detector |estatística linear| com
ruído gaussiano e limiar no falso alarme, P = Φ(B·f/s − z_c) + Φ(−B·f/s − z_c), com um só
parâmetro livre. Para a D²NN óptica, a forma também vale: o teste da razão de verossimilhança
contra uma potência livre (u = (B·f/s)^β) não rejeita β = 1 em nenhum dos 33 casos (p > 0,01)
e dá β mediano de 1,03, igual ao do filtro (1,04). Com a máscara de fase de Fourier, a energia
na janela é dominada pelo pico de correlação, isto é, |estatística linear|².

### 4.6 Inclinação do campo (script 71)

Para θ = 0°, 30°, 45°, 60° e 75°, campo total B/cosθ: B_los·f e a realização do ruído ficam
iguais aos da varredura do 54, e só a física muda (I com θ, ruído ∝ I no CRISP). Razão
B_los·f₅₀(θ)/B_los·f₅₀(0) = **1,000** em todos os θ, instrumentos e detectores (critério:
1 ± 0,05). Os valores em θ = 0 (31,1/29,6, 50,2/57,8 e 10,3/8,6 G) reproduzem o 54.

- É verificação de consistência: o motor físico tem V ∝ cosθ por construção, e com o
  desdobramento Zeeman pequeno o I quase não muda com θ. O que o script acrescenta é que a D²NN,
  treinada só com θ = 0, respeita a escala.
- As Células 33/33B do Colab usavam cos²θ (incorreto para V).
- Uma primeira montagem (mesmo B em todo θ) comprimia a faixa de B_los·f (até 39 G em 75°) e
  deixava o estimador na borda; foi descartada antes de rodar.

### 4.7 Origem da perda no ViSP (script 73)

Só detecção (5% FA), máscaras calculadas, sem treino; 3500 perfis do ViSP (sementes do 54);
energia exata em k pelo campo na janela, que é linear em V. Razão B·f₅₀ / filtro |a|:

| Máscara | Mediana em k | k = 1 [IC 95%] |
|---|---|---|
| 67 (POF médio, fase treinada, ±200 pm) | 1,42 | 1,32 [1,23; 1,40] |
| POF médio ±200 pm, sem treino | 1,97 | 1,86 [1,72; 2,00] |
| POF do ViSP, sem diafragma / ±200 / ±120 / ±80 pm | 2,09 / 1,63 / 1,60 / 1,56 | 2,10 / 1,63 / 1,56 / 1,56 |
| **Casado complexo do ViSP (H = S*), sem diafragma** | **0,99** | **1,00 [0,99; 1,00]** |
| Casado complexo do ViSP, ±200 pm | 1,03 | 1,03 [0,99; 1,06] |

- **Decisão (definida antes): H2** — a perda é da máscara só de fase. A H1 (máscara única) não
  explica: o POF calculado para o próprio ViSP não passa de 1,56.
- **Mecanismo:** o POF tem ganho 1 em todas as frequências; no ViSP o ruído cobre ±5 Å e a
  máscara não atenua as frequências em que o sinal é fraco. H = S* faz exatamente o filtro
  casado: a óptica com a máscara complexa *é* o filtro casado.
- **O treino de fase compensa parte:** de 1,97 (POF médio sem treino) para 1,42 (67), melhor que
  qualquer POF fixo do ViSP.

### 4.8 D²NN com máscara de amplitude e fase (scripts 74 e 54)

`D2NNOpticaComplexa`: t = a·e^{iφ}, 0 < a < 1 (passiva: SLM de fase + máscara de absorção),
iniciada em H = S*/max|S| do template médio; uma máscara para os três instrumentos; mesmo
protocolo de treino do 62/67. Teste grande v2 e B·f₅₀ em k = 1 (estimador físico):

| Variante | Acurácia | IBIS | CRISP | ViSP | Média | Transmissão ⟨a²⟩ |
|---|---|---|---|---|---|---|
| filtro \|a\| | 86,13% | 29,6 G | 57,8 G | 8,6 G | | |
| 67 (só fase, ±200 pm) | 86,03% | 1,05 | **0,87** | 1,20 | 1,04 | 100% |
| complexa fixa (H = S*) | 86,40% | 0,98 | 1,00 | 0,97 | 0,98 | 0,7% |
| **complexa treinada (escolhida)** | **86,73%** | 1,05 | 0,95 | **1,02** | 1,01 | 3,1% |
| complexa treinada ±200 pm | 86,53% | 1,01 | 0,93 | 1,08 | 1,01 | 2,8% |

- **Critérios (definidos antes):** C1, ViSP ≤ 1,10: 1,02, **atendido**. C2, IBIS ≤ 1,10 e CRISP
  ≤ 67 + 0,05 = 0,92: CRISP 0,95, **não atendido** (por 0,03).
- **Curva completa (54 com o 74)**, razão 74 / filtro |a| de k = 0,1 a 2 (valores arredondados
  da tabela do 54): IBIS mediana 1,01 (1,00–1,07), CRISP **0,89** (0,84–0,98), ViSP **1,00**
  (0,96–1,14; os extremos em k ≤ 0,3, onde o B·f₅₀ é de 1–2 G e o arredondamento pesa). Com o
  67: 1,05 / 0,82 / 1,38.
- **Acurácia por instrumento em k = 1 (74 / 67 / filtro):** IBIS 86,7 / 85,9 / 87,7%; CRISP
  79,0 / 80,4 / 78,4%; **ViSP 97,2 / 94,4 / 91,8%** (Rede no ViSP: 92,0 / 84,7 / 78,1%).
- **Planejamento do CRISP:** média de 11 varreduras, 16,1 G (67: 14,8 G; filtro 19,2 G).
- **Leitura:** o CRISP perde parte da vantagem do 67 (0,82 → 0,89). Hipótese (não testada): o
  ganho no CRISP vem de ponderar o ruído ∝ I, uma ponderação ao longo de λ no plano de entrada,
  que uma máscara no plano de Fourier só imita; com a fase livre o 67 achou essa aproximação,
  e o treino da complexa, que parte do filtro casado, ficou perto dele.
- **Transmissão:** ⟨a²⟩ na pupila é 0,7% (fixa) e 3,1% (treinada). Com o ruído na entrada, como
  no modelo, isso não muda a detecção; num instrumento com ruído no detector é custo de luz. A
  fração da potência do sinal que passa é maior (a amplitude é alta onde o sinal está): medida
  no script 75 (seção 4.9), 24–44%.

### 4.9 Referências para o artigo B: ótimo branqueado, frações e quantização (script 75)

Sem treino. Mesmos conjuntos e sementes do 54/70/73 (sem campo 1500; varredura Quieta + Rede,
1000/classe; mistura de instrumentos). Perfis gerados com a cadeia de
`generate_degraded_profiles` guardando também a grade nativa (conferido contra a função).
Estimador físico (β = 1); D²NN lidas pela energia na janela (como no 73); IC 95% por bootstrap
pareado em k = 1.

**A. Ótimo branqueado.** Filtro casado na grade nativa com pesos 1/σ_k² do perfil sem ruído,
a_w = Σ t_k V_k/σ_k² / Σ t_k²/σ_k². No IBIS e no ViSP (σ constante) coincide com o filtro
comum (controle).

| Razão: mediana em k (faixa); k = 1 [IC 95%] | IBIS | CRISP | ViSP |
|---|---|---|---|
| ótimo / filtro \|a\| | 0,99 (0,97–1,02); 1,00 [0,96; 1,03] | **0,79 (0,75–0,86); 0,82 [0,75; 0,90]** | 1,00; 1,00 |
| 67 / ótimo | 1,07 (1,02–1,16); 1,05 [0,95; 1,14] | **1,05 (1,00–1,09); 1,06 [0,98; 1,14]** | 1,38 (1,20–1,51); 1,20 [1,01; 1,44] |
| 74 / ótimo | 1,03 (1,00–1,06); 1,06 [1,00; 1,13] | 1,15 (1,09–1,22); 1,15 [1,08; 1,26] | 1,00 (0,97–1,14); 1,02 [0,96; 1,09] |

(B·f₅₀ em k = 1, ótimo / filtro: IBIS 29,5 / 29,6 G; CRISP **47,5 / 57,8 G**; ViSP 8,6 / 8,6 G.
As razões 67/filtro e 74/filtro reproduzem o 54: medianas 1,05 / 0,82 / 1,38 e 1,02 / 0,92 / 1,00.)

- **Critério A (definido antes):** a D²NN 67 atinge o ótimo no CRISP se 67/ótimo em k = 1 tiver
  IC contendo 1 ou for ≤ 1,05. Resultado 1,06 [0,98; 1,14]: **atendido pelo IC**; a estimativa
  pontual fica 6% acima. Redação: estatisticamente indistinguível do ótimo em k = 1, a ~5% dele
  na mediana em k.
- **Fração do ganho recuperada pelo 67:** (57,8 − 50,2)/(57,8 − 47,5) = 74%.
- **O 74 não branqueia:** parte de H = S* (casado sem branqueamento) e o treino não o tira de lá
  (coerente com o C2 reprovado no 74).

**B. Frações da energia de entrada** (antes do diafragma). Sinal: template do instrumento
(B·f = 100 G). Ruído: 300 realizações sem campo por instrumento.

| Máscara | Instrumento | Sinal passa | Sinal na janela | Ruído passa | Ruído na janela |
|---|---|---|---|---|---|
| 67 (só fase) | IBIS | 99,4% | 1,17% | 72,4% | 0,086% |
| | CRISP | 98,1% | 0,98% | 84,8% | 0,128% |
| | ViSP | 91,0% | 0,99% | 8,3% | 0,010% |
| 74 (amp. e fase) | IBIS | 35,8% | 0,093% | 6,4% | 0,0061% |
| | CRISP | 24,4% | 0,074% | 9,6% | 0,0103% |
| | ViSP | 44,4% | 0,127% | 1,6% | 0,0006% |

- Transmissão média ⟨a²⟩ do 74 na pupila: 3,08%. A absorção cai sobretudo fora do sinal.
- No 67 a perda de sinal é só da pupila e do diafragma; no ViSP o diafragma corta 92% do ruído.
- O 74 leva ~10× menos energia de sinal à janela que o 67. Com o ruído na entrada (o modelo) isso
  não altera a detecção; só pesa se o ruído do detector dominar.
- A razão sinal/ruído entre colunas não é a SNR de detecção (normalizações diferentes).

**C. Quantização** (fase em 2ⁿ níveis uniformes em [−π, π), sem retreino; no 74 também a
amplitude em 2ⁿ níveis em [0, 1], só reportada). Razão B·f₅₀(quantizada)/B·f₅₀(contínua) em
k = 1, pior instrumento:

| Bits | 67 fase | 74 fase | 74 fase + amplitude |
|---|---|---|---|
| 8 | 1,000 | 1,000 | 1,000 |
| 6 | 1,000 | 1,000 | 1,000 |
| 5 | 1,016 | 1,000 | 1,000 |
| 4 | 1,000 | 1,000 | 1,000 |
| 3 | 1,021 | 1,012 | 1,002 |
| 2 | **1,154** (ViSP) | 0,977 | 1,021 |

- **Critério C (definido antes):** menor n com perda ≤ 5% para todo n' ≥ n nos três instrumentos.
  **67: 3 bits. 74: ≤ 2 bits** (o piso não foi encontrado na faixa testada).
- Não inclui fill factor, crosstalk entre pixels nem erro de calibração do SLM.
- Figura: `cel75_otimo_branqueado.png` (razões contra o ótimo em função de k).

### 4.10 Máscara de amplitude e fase projetada como filtro branqueado (script 76)

Pergunta: a falta de branqueamento do 74 no CRISP é limite da arquitetura ou do treino? Num 4f
o campo no pico de correlação é ⟨h, X⟩, com h a resposta impulsional da máscara; uma máscara de
amplitude e fase realiza qualquer peso. Teste só de detecção, sem treino (como o 73), com
máscaras calculadas H = conj(F{X_w})/max|F{X_w}| na pupila, sem diafragma, janela 3 × 5.

- **Peso branqueado exato na grade LAM:** o ótimo nativo é u·V_nat, u = t/σ² (σ do Sol quieto
  degradado). Com V_LAM = M·V_nat (M = interpolação linear), h = M(MᵀM)⁻¹u satisfaz Mᵀh = u, então
  h·V_LAM = u·V_nat. Um peso ingênuo t/σ² na grade LAM não serve: a interpolação pondera cada
  ponto nativo pelo espaçamento local, e a grade do CRISP é irregular (visto no teste reduzido).
- **Máscara única:** w = Σ_k p_k·w_k·‖t_k‖/‖w_k‖ (com σ constante reduz à mistura de templates
  do 74).

| Razão: mediana em k (faixa); k = 1 [IC 95%] | IBIS | CRISP | ViSP |
|---|---|---|---|
| casado S* / filtro \|a\| (controle) | 1,01 (1,00–1,02); 1,01 [1,00; 1,05] | 1,00 (0,96–1,02); 1,01 [0,99; 1,02] | 1,00 (0,97–1,02); 0,97 [0,89; 1,00] |
| branqueado LAM (digital) / ótimo | 1,00; 1,00 | 1,00; 1,00 | 1,00; 1,00 |
| filtro \|a\| / ótimo | 1,01 (0,98–1,03); 1,00 | **1,27 (1,16–1,33); 1,22 [1,13; 1,32]** | 1,00; 1,00 |
| casado S* / ótimo | 1,02; 1,02 [0,98; 1,08] | 1,26 (1,17–1,30); 1,22 [1,12; 1,34] | 1,00; 0,97 [0,91; 1,00] |
| **branqueado (instrumento) / ótimo** | **1,02 (0,99–1,03); 1,00 [0,96; 1,03]** | **1,00 (0,98–1,04); 1,01 [1,00; 1,03]** | **1,00 (0,97–1,02); 0,97 [0,91; 1,00]** |
| branqueado único (3 instr.) / ótimo | 1,03 (0,97–1,09); 1,02 [0,96; 1,08] | 1,11 (1,05–1,17); 1,10 [1,03; 1,21] | 1,03 (0,89–1,08); 0,92 [0,83; 1,02] |

(B·f₅₀ em k = 1 no CRISP: máscara branqueada 48,0 G; ótimo 47,5 G; casado S* 58,1 G; filtro
|a| 57,8 G; máscara única 52,5 G. Transmissão ⟨|H|²⟩ na pupila: 0,5–0,8% nos casados, 1,39% no
branqueado do CRISP, 0,78% na única.)

- **Critérios (definidos antes):** controle (casado / filtro em 1 ± 0,03): **atendido**
  (1,01 / 1,00 / 1,00). W1 (branqueado LAM / ótimo no CRISP ≤ 1,03): **atendido** (1,00): o peso
  fixo com o σ do Sol quieto não perde nada para o σ de cada perfil. W2 (máscara branqueada /
  ótimo no CRISP ≤ 1,05): **atendido** (1,00). W3 (máscara única, só reportado): 1,03 / 1,11 / 1,03.
- **Leitura:** a óptica com a máscara branqueada *é* o detector ótimo, nos três instrumentos. A
  janela 3 × 5 e a codificação da entrada não perdem nada (controle).
- **Treino contra projeto:** a 74 treinada a partir de S* fica em 1,15 no CRISP; a 67 só de fase
  treinada, em 1,05; a máscara única projetada, em 1,11; a projetada por instrumento, em 1,00.

## 5. Achados

(1–4 em `RECONSTRUCAO_NORMALIZACAO.md`; 5–10 em `RETREINO_ENTRADA_OBSERVAVEL.md`; 11–15 em
`ROBUSTEZ_E_DETECCAO.md`; 16–21 e 32 em `MODELO_INSTRUMENTAL.md`.)

22. **Nos modelos híbridos (47, 57, 59) a óptica não contribui** (seção 1). A cabeça eletrônica
    faz toda a classificação; as três `CamadaFase` recebem o mesmo gradiente e ficam idênticas.
    Uma "D²NN" com uma cabeça eletrônica grande não demonstra computação óptica.
23. **Com classes de mesma forma, toda leitura óptica se reduz a um escalar, e uma máscara de
    fase com um detector atinge o filtro casado** (seções 2 e 3). A D²NN puramente óptica, com
    6 parâmetros eletrônicos, empata com o filtro |a| e supera o híbrido de 4,2 M parâmetros
    em 10 pp. Explica o achado 19: a perda de eficiência estava na cabeça eletrônica.
24. **Uma máscara de fase pura não atenua o ruído fora da linha; um diafragma de campo resolve**
    (seção 4). É a pendência 3 do `MODELO_INSTRUMENTAL.md` (janela de entrada) resolvida por
    um elemento óptico passivo.
25. **No CRISP a D²NN óptica supera o filtro |a| e se aproxima do ótimo ponderado** (seção 4.3),
    porque o treino aprende a ponderação do ruído proporcional a I (achado 17). Com o estimador
    físico: razão 0,87 [0,81; 0,94] em k = 1 e 50,2 G contra 49,3 G do ótimo. *Revisto no
    achado 33: o ótimo branqueado nos mesmos perfis é 47,5 G.*
26. **O desvio Doppler custa o mesmo à D²NN óptica e ao filtro de template fixo** (seção 4.4):
    ≤ 2% até ±3 km/s e 5–11% em 5 km/s. Com a janela fixa, a leitura óptica é a correlação no
    atraso zero, a mesma do filtro. Não é preciso uma fileira de detectores.
27. **O B·f₅₀ deve ser estimado com a forma física da curva de detecção** (seção 4.5;
    `ROBUSTEZ_E_DETECCAO.md`, seção 6.6). A logística em log B·f tem forma errada (viés de
    −11% acima de 15 G); sem forma (isotônica), a dispersão sobe para ~9%. A forma
    Φ(B·f/s − z_c) + Φ(−B·f/s − z_c), com um parâmetro, passa na validação contra a teoria.
28. **A curva de detecção da D²NN óptica tem a forma de um detector linear** (β = 1,03; seção
    4.5). A D²NN de uma máscara se comporta, no critério de detecção, como um filtro casado
    óptico: um argumento direto para o artigo B.
29. **Os limites de detecção valem para B_los·f = B·f·cosθ** (seção 4.6; razão 1,000 até 75°).
30. **A perda da D²NN óptica no ViSP é da máscara só de fase, não da máscara única** (seção
    4.7). Uma máscara de amplitude e fase H = S* no plano de Fourier reproduz exatamente o filtro
    casado (0,99–1,00×).
31. **Uma única máscara de amplitude e fase atinge o filtro casado nos três instrumentos**
    (seção 4.8): 1,01 / 0,89 / 1,00 (IBIS / CRISP / ViSP), acurácia 86,73%. A máscara só de
    fase treinada ganha mais no CRISP (0,82) e perde no ViSP (1,38): as duas arquiteturas são
    complementares. Custo: transmissão de 3%.
33. **A máscara só de fase treinada aprende o branqueamento; a de amplitude e fase, não**
    (seção 4.9). O filtro casado branqueado (ótimo com ruído ∝ I) dá 47,5 G no CRISP em k = 1,
    contra 57,8 G do filtro |a|. A D²NN 67 fica em 1,06 [0,98; 1,14] do ótimo (critério A
    atendido pelo IC) e recupera 74% do ganho do branqueamento; a 74 fica em 1,15 [1,08; 1,26].
    Com ruído branco (IBIS, ViSP) a 74 atinge o ótimo (1,03 / 1,00). Complementaridade revista:
    amplitude e fase resolvem o ruído fora da linha; só fase aprende a ponderação ∝ I.
34. **A máscara de amplitude é seletiva** (seção 4.9): transmite 24–44% da energia do sinal e
    1,6–9,6% da do ruído, com transmissão média de 3%. A perda de luz não é um argumento contra
    ela enquanto o ruído da entrada dominar.
35. **A quantização de fase do SLM não é limitante** (seção 4.9): 3 bits bastam para o 67 e
    2 bits para o 74 (perda ≤ 5%), sem retreino. Um SLM comercial de 8 bits tem folga grande.
36. **Uma máscara de amplitude e fase projetada como filtro branqueado realiza o detector ótimo
    em qualquer estrutura de ruído** (seção 4.10): 1,00 do ótimo nos três instrumentos, incluindo
    o CRISP (ruído ∝ I), onde o filtro |a| e o casado S* ficam em 1,22–1,27. A falta de
    branqueamento do 74 (achado 33) é do treino, que parte de S*, e não da arquitetura. Uma
    máscara única para os três custa 3 / 11 / 3%; com um SLM reprogramável, a máscara por
    instrumento é gratuita. Revisa a leitura do achado 33: a complementaridade "só fase × amplitude
    e fase" vale para as máscaras *treinadas*; a projetada resolve os dois casos.

## 6. Pendências

1. ~~Doppler~~ — feito (script 68, seção 4.4; achado 26).
2. ~~Máscara por instrumento para o ViSP~~ — a causa era a máscara só de fase; resolvido com
   amplitude e fase (seções 4.7 e 4.8). O ganho do CRISP com amplitude e fase é obtido pela
   máscara branqueada projetada (script 76, achado 36). Opcional: treinar a 74 partindo da máscara
   branqueada (se um revisor pedir).
3. **k sorteado no treino** para o ponto de operação acompanhar o ruído (k > 1).
4. ~~Filtro casado branqueado como referência explícita~~ — feito (script 75, seção 4.9;
   achado 33).
5. **Realizabilidade física:** a quantização de fase e a fração de sinal da máscara de amplitude
   foram feitas (script 75; achados 34 e 35). Falta, só no texto: a implementação óptica da
   entrada X_k = coef_k·V (rampa de 64 linhas, subtração da média; modulação e polarimetria) e
   a discussão de ruído na entrada contra ruído no detector.
6. **Aplicação a dados reais** quando chegarem os cubos do SST (scripts 63–66 prontos) e os λ
   reais do CRISP (retreino na grade real).

## 7. Reprodutibilidade

| Script | Tempo (RTX 4060) | Comando |
|---|---|---|
| 61 | ~5 min | `python Scripts/61_ablacao_optica.py` |
| 62 | ~16 min | `python Scripts/62_d2nn_optica_pura.py` |
| 67 | ~14 min | `python Scripts/67_d2nn_optica_diafragma.py` |
| 54 (62 ou 67) | 6–13 min | `python Scripts/54_snr_detection.py --instrumental v2 --modelo 67` |
| 68 | ~2 min | `python Scripts/68_doppler.py` |
| 69 | ~4 min | `python Scripts/69_validacao_estimador.py` |
| 70 | ~12 min | `python Scripts/70_estimador_fisico.py` |
| 71 | ~1 min | `python Scripts/71_inclinacao.py` |
| 73 | ~7 min | `python Scripts/73_mascara_visp.py` |
| 74 | ~15 min | `python Scripts/74_d2nn_complexa.py` |
| 54 (74) | ~13 min | `python Scripts/54_snr_detection.py --instrumental v2 --modelo 74` |
| 75 | ~11 min | `python Scripts/75_referencias_artigo_b.py` |
| 76 | ~4 min | `python Scripts/76_mascara_branqueada.py` (usa funções do 75) |

Checkpoints: `modelo_d2nn_optico_pof_fixo.pt`, `modelo_d2nn_optico_pof_treinado.pt` (62),
`modelo_d2nn_optico_diafragma_{200,120,80}pm.pt` e `modelo_d2nn_optico_v2.pt` (67, cópia do
±200 pm); `modelo_d2nn_complexa_fixa.pt`, `modelo_d2nn_complexa_treinada.pt`,
`modelo_d2nn_complexa_treinada_200pm.pt` e `modelo_d2nn_complexo_v2.pt` (74, cópia da treinada sem
diafragma). Lidos por `carregar_modelo` (arquiteturas `D2NNOptica` e `D2NNOpticaComplexa`). Os
scripts 62, 67 e 74 estão no `PULAR` do `00_run_all.py`; 61, 68–73 rodam nele.
