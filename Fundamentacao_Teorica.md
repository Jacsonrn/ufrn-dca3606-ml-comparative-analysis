# 🧠 Fundamentação Teórica

Este documento centraliza a teoria matemática e estratégica por trás de todas as decisões do código, **organizado estritamente pelo cronograma de dias de desenvolvimento do projeto**.
---

# 📅 DIA 1: Seleção dos Datasets e Configuração do Ambiente

Neste dia, estabelecemos a fundação do projeto, configurando a infraestrutura virtual (`.venv`) e selecionando as bases de dados que determinariam a complexidade matemática dos dias seguintes.

### 📚 Fundamentação Teórica do Dia 1
A escolha dos dados define o teto de sucesso do treinamento. Optamos por **Breast Cancer** (Classificação) e **California Housing** (Regressão) porque ambos possuem volume amostral estatisticamente significativo (≥ 500 observações), não possuem dados faltantes nativamente, e contêm 100% de *features* (variáveis explicativas) numéricas contínuas. 

Essa natureza estritamente numérica evita a *Maldição da Dimensionalidade* que ocorreria se precisássemos aplicar *One-Hot Encoding* em múltiplas variáveis categóricas (texto). Assim, nossa equipe manteve a complexidade e o tempo de computação focados apenas onde o projeto exige: no estudo de desempenho e nas métricas dos 12 modelos de Machine Learning.

### ❓ 10 Perguntas sobre o Dia 1
1. Por que vocês escolheram o dataset Breast Cancer especificamente para a tarefa de classificação? O que o torna didático?
2. Quais são as variáveis-alvo (labels/targets) de cada dataset e de qual tipo estatístico elas são?
3. Vocês chegaram a considerar o uso de datasets com dados textuais ou de imagens? Por que priorizaram dados estruturados numéricos?
4. Qual a dimensão exata (linhas e colunas) dos datasets escolhidos no momento do carregamento? O que esses números representam na álgebra linear do modelo?
5. Considerando o California Housing, qual é a origem formal desta base de dados (Kaggle, UCI, scikit-learn)?
6. O dataset de classificação é balanceado? Qual a proporção exata entre tumores benignos e malignos e como vocês descobriram isso?
7. Como a natureza espacial e não-linear do California Housing beneficia e enriquece a nossa escolha de treinar algoritmos como Random Forest e MLP?
8. O limite mínimo exigido de 500 amostras é realmente suficiente para treinar algoritmos gulosos por dados, como as Redes Neurais Artificiais (MLP)?
9. Existe algum risco de vazamento de dados (*data leakage*) apenas pelas variáveis que vieram na coleta original desses dados?
10. Se a variável alvo do California Housing (preço) fosse convertida em "Caro" e "Barato", nós estaríamos resolvendo um problema de regressão ou classificação?

---

# 📅 DIA 2: Análise Exploratória (EDA) e Pré-Processamento

Neste dia, transformamos os dados brutos. Primeiro, investigamos as anomalias visuais e estatísticas (EDA). Em seguida, convertemos essa matriz de forma matemática (Pré-Processamento) para que os algoritmos conseguissem absorvê-la sem distorções de escala.

### 📚 Fundamentação Teórica do Dia 2

**1. Sobre a Análise Exploratória (EDA)**
A EDA não é apenas sobre desenhar gráficos coloridos, mas sobre a descoberta de anomalias que quebram premissas matemáticas dos modelos. O uso da função `.describe()` validou a diferença drástica de escalas de grandeza entre as colunas. A plotagem do *Boxplot* revelou a presença estrutural de *outliers* na coluna de renda (MedInc). Já o *Countplot* atestou um leve desbalanceamento no dataset de Câncer (62% benigno vs 38% maligno), o que exigirá muita atenção à métrica de *Recall* no Dia 5, já que errar um diagnóstico maligno (falso-negativo) é fatal.

**2. Sobre o Pré-Processamento e Padronização**
* **Train/Test Split:** Aplicamos uma separação de 80/20 para garantir o Método *Holdout*, reservando dados virgens para testar o poder de generalização do modelo e evitar o temido *Overfitting* (memorização). 
* **Estratificação (Stratify):** Usamos o parâmetro `stratify` na classificação para garantir que os conjuntos de Treino e Teste preservem rigorosamente a proporção biológica de 62/38, impedindo que, por obra do acaso matemático, a nossa matriz de teste recebesse apenas pacientes benignos.
* **StandardScaler (Normalização Z-Score):** Transforma os dados forçando-os a ter média 0 e desvio padrão 1. É mandatório para modelos que calculam distância geométrica (K-Means, SVM) e otimização por Gradiente Descendente (Redes Neurais). Ele impede que variáveis milionárias anulem o peso matemático de variáveis de baixa magnitude.

### ❓ 10 Perguntas a Análise Exploratória (EDA)
1. O que é um *outlier* (ponto fora da curva) e como ele é visivelmente identificado no gráfico de Boxplot que vocês plotaram?
2. O que representa, matematicamente falando, a "caixa" central de um Boxplot? (Resposta esperada: O Intervalo Interquartil - IQR, que vai do percentil 25% ao 75%).
3. Vocês notaram desbalanceamento nas classes do Breast Cancer. Como a matemática dos modelos lida com classes desbalanceadas?
4. A Acurácia seria uma boa métrica de avaliação se o dataset fosse 99% benigno e apenas 1% maligno? Por quê?
5. Qual a diferença técnica entre o Histograma (usado para ver o preço das casas) e um Gráfico de Barras simples?
6. O código acusou a existência de valores nulos ou faltantes? Como a equipe trataria (imputação) se eles existissem?
7. O que acontece com a linha matemática da Regressão Linear Simples se nós não removermos os outliers severos encontrados no Boxplot?
8. Durante a geração das estatísticas descritivas (`.describe()`), como vocês detectam qual variável tem a maior variância?
9. O que significa dizer que um histograma tem um "viés à direita" (right-skewed distribution), como vimos nos preços das casas?
10. Os algoritmos baseados em árvores (Decision Tree, Random Forest) sofrem muito impacto negativo por causa dos outliers que vimos nos Boxplots?

### ❓ 10 Perguntas o Pré-Processamento
1. Por que é estritamente obrigatório dividir os dados em treino e teste antes de fazer qualquer modelagem? O que é um modelo em *Overfitting*?
2. Vocês utilizaram `stratify` apenas na divisão do dataset de câncer. O que isso faz no código e por que é vital para não distorcer o teste?
3. Por que vocês NÃO usaram a estratificação na hora de dividir o California Housing? (Resposta esperada: porque variáveis contínuas numéricas não formam classes).
4. O que o algoritmo `StandardScaler` faz matematicamente com os valores de uma coluna de dados? Qual é a fórmula aplicada?
5. O que aconteceria com o cálculo da distância Euclidiana do K-Means se vocês esquecessem de aplicar o StandardScaler antes?
6. Pergunta crítica: Por que o comando `scaler.fit()` deve ser rodado APENAS no conjunto de treino (`X_train`), e nunca no conjunto de teste ou na base inteira? (Resposta esperada: Data Leakage).
7. Já que modelos como Random Forest e XGBoost são imunes a diferenças de escalas, por que vocês padronizaram os dados de regressão mesmo assim?
8. Existem alternativas ao `StandardScaler` (como o `MinMaxScaler`). Em qual cenário o MinMax seria melhor que o Standard?
9. O StandardScaler altera o formato visual e original da distribuição dos dados (o "desenho" do histograma) ou ele altera apenas a métrica dos eixos X e Y?
10. Vocês definiram o tamanho do conjunto de teste em 20%. Vocês acham que testar apenas 114 amostras finais de câncer é um volume seguro para garantir eficácia médica global do algoritmo?
