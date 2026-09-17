# 🧠 Fundamentação Teórica e Preparação para Defesa

Este documento centraliza a teoria matemática e estratégica por trás de todas as decisões do código, **organizado estritamente pelo cronograma de dias de desenvolvimento do projeto**. Ele serve como um guia definitivo de estudo para responder às perguntas da banca examinadora durante a apresentação oral.

---

# 📅 DIA 1: Seleção dos Datasets e Configuração do Ambiente

Neste dia, estabelecemos a fundação do projeto, configurando a infraestrutura virtual (`.venv`) e selecionando as bases de dados que determinariam a complexidade matemática dos dias seguintes.

### 📚 Fundamentação Teórica do Dia 1

**1. Aprendizado Supervisionado vs. Não Supervisionado**
O Machine Learning se divide em dois grandes paradigmas. No **Aprendizado Supervisionado**, o algoritmo recebe um conjunto de dados onde cada exemplo possui uma "resposta correta" associada (o rótulo, ou *label*). O modelo aprende a mapear entradas para saídas observando esses pares (entrada, resposta). No **Aprendizado Não Supervisionado**, o algoritmo recebe apenas os dados brutos, sem nenhum gabarito, e precisa descobrir padrões ocultos por conta própria (agrupamentos, redução de dimensionalidade). A escolha dos datasets para este projeto precisou atender a ambos os paradigmas: os rótulos servem para treinar os 12 modelos supervisionados, e os mesmos atributos numéricos (sem os rótulos) serão usados para alimentar PCA, K-Means e DBSCAN no Dia 3.

**2. Classificação vs. Regressão: A Natureza da Variável Alvo**
Dentro do Aprendizado Supervisionado, a natureza estatística da variável alvo (*target*) define se o problema é de **Classificação** ou de **Regressão**:
* **Classificação:** A variável alvo é **categórica** (discreta). Ela pertence a um conjunto finito de classes. No Breast Cancer, o alvo é binário: `0 = Maligno` ou `1 = Benigno`. O modelo precisa aprender a "decidir" a qual classe cada nova amostra pertence. Quando existem apenas duas classes, chamamos de **Classificação Binária**; quando existem três ou mais, chamamos de **Classificação Multiclasse**.
* **Regressão:** A variável alvo é **numérica contínua**. Ela pode assumir infinitos valores dentro de um intervalo real. No California Housing, o alvo é o preço mediano das casas (`MedHouseVal`), que varia continuamente (ex: 1.234, 3.500, 5.001). O modelo precisa aprender a "prever um número", não a escolher uma categoria.

Esta distinção é absolutamente vital porque ela determina quais algoritmos, quais funções de perda (*loss functions*) e quais métricas de avaliação são matematicamente válidas. Aplicar uma métrica de classificação (como Acurácia) num problema de regressão é um erro conceitual grave, e vice-versa.

**3. Critérios de Seleção dos Datasets**
A escolha dos dados define o teto de sucesso do treinamento. Optamos por **Breast Cancer Wisconsin** (Classificação) e **California Housing** (Regressão) com base nos seguintes critérios técnicos rigorosos:
* **Volume Amostral:** Ambos possuem volume estatisticamente significativo (569 e 20.640 amostras, respectivamente), superando com folga o mínimo de 500 exigido pela disciplina. Volumes maiores reduzem a variância nas estimativas de desempenho e permitem que os modelos capturem padrões mais robustos.
* **Ausência de Dados Faltantes (*Missing Values*):** Nenhum dos dois datasets contém valores nulos. Isso elimina a necessidade de técnicas de imputação (como substituir valores ausentes pela média ou mediana), que introduziriam ruído artificial e adicionariam uma camada de complexidade desnecessária ao escopo deste projeto.
* **Natureza 100% Numérica das Features:** Todas as variáveis explicativas (colunas) são números reais (`float64`). Isso é estratégico porque evita a necessidade de aplicar técnicas de codificação categórica.

**4. A Maldição da Dimensionalidade e o One-Hot Encoding**
Se tivéssemos escolhido um dataset com variáveis categóricas (exemplo: uma coluna "Cor" com valores "Vermelho", "Azul", "Verde"), precisaríamos convertê-las em números usando uma técnica chamada **One-Hot Encoding**. Essa técnica cria uma nova coluna binária (0 ou 1) para cada categoria existente. Se a coluna "Cidade" tivesse 200 cidades diferentes, o One-Hot Encoding adicionaria 200 novas colunas à nossa matriz de dados. Esse fenômeno é a **Maldição da Dimensionalidade** (*Curse of Dimensionality*): quando o número de colunas cresce descontroladamente, o espaço geométrico fica tão vasto e esparso que os algoritmos de distância (como K-Means e KNN) perdem completamente a capacidade de medir proximidade entre pontos, e o tempo de computação explode exponencialmente.

Ao escolher datasets estritamente numéricos, mantivemos a complexidade e o tempo de computação focados apenas onde o projeto exige: no estudo de desempenho e nas métricas dos 12 modelos de Machine Learning.

**5. Datasets como Standard Benchmarks**
Breast Cancer Wisconsin e California Housing são **datasets de referência padrão** (*standard benchmarks*) na comunidade global de Ciência de Dados. O criador da biblioteca `scikit-learn` os embutiu diretamente no código-fonte da ferramenta (`load_breast_cancer()` e `fetch_california_housing()`), precisamente para que engenheiros e pesquisadores testem algoritmos sem depender de downloads externos. Isso garante **reprodutibilidade** total: qualquer pessoa em qualquer computador do mundo, ao rodar nosso código, obterá os mesmos dados e os mesmos resultados numéricos.

**6. Configuração do Ambiente Virtual (`.venv`)**
Um **ambiente virtual** é um diretório isolado que contém uma instalação independente do Python e de todas as suas bibliotecas. Ele resolve um problema crítico de engenharia de software: o **conflito de dependências**. Se o seu sistema operacional usa o Python 3.8 com a versão 0.24 do `scikit-learn`, mas o nosso projeto precisa da versão 1.3, instalar a versão nova globalmente poderia quebrar outros programas do seu computador. O `.venv` isola completamente essas versões, e o arquivo `requirements.txt` garante que qualquer colega de equipe instale exatamente as mesmas versões, garantindo **reprodutibilidade** do ambiente.

**7. O Papel do `random_state=42`**
Em diversos pontos do código, usamos o parâmetro `random_state=42`. Algoritmos como `train_test_split` e `KMeans` dependem de números aleatórios internamente (para embaralhar dados, para inicializar centroides). Se não fixássemos essa semente, cada execução do código produziria resultados ligeiramente diferentes, impossibilitando a comparação justa entre modelos. Fixando a semente em 42 (um número arbitrário, escolhido por convenção da comunidade), garantimos que toda execução reproduza exatamente o mesmo embaralhamento e as mesmas inicializações.

### ❓ 10 Perguntas da Banca sobre o Dia 1
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

**1. Análise Exploratória de Dados (EDA) — Conceito e Objetivo**
A *Exploratory Data Analysis* (EDA) é uma disciplina estatística formalizada por John Tukey na década de 1970. Seu objetivo é investigar os dados antes de aplicar qualquer modelo, usando ferramentas visuais e numéricas para descobrir: a distribuição estatística de cada variável, a existência de valores atípicos (outliers), a presença de correlações entre variáveis, e se há desbalanceamento entre as classes no caso de classificação. A EDA não é uma etapa opcional ou cosmética — ela revela informações que, se ignoradas, podem fazer um modelo aparentemente "perfeito" falhar catastroficamente na realidade.

**2. Estatísticas Descritivas (`.describe()`)**
A função `.describe()` do pandas calcula, para cada coluna numérica, um resumo composto por 8 valores fundamentais:
* **count:** Número de valores não-nulos (confirma a integridade dos dados).
* **mean (Média):** A soma de todos os valores dividida pela contagem. Indica o "centro de gravidade" da distribuição.
* **std (Desvio Padrão):** Mede a dispersão dos valores em torno da média. Um desvio padrão alto significa que os valores estão muito espalhados; um desvio baixo significa que estão concentrados. É a raiz quadrada da **variância**.
* **min / max:** Os valores extremos da coluna. Permitem identificar imediatamente se existe algum valor absurdo (ex: uma idade de -5 anos indicaria um erro de digitação).
* **25% / 50% / 75% (Quartis):** Dividem os dados ordenados em quatro partes iguais. O 50% é a **mediana**, que é mais robusta que a média na presença de outliers (uma única casa de R$ 100 milhões distorce a média, mas não a mediana).

Ao analisar essas estatísticas, descobrimos que as features dos nossos datasets estão em escalas de grandeza drasticamente diferentes: no California Housing, `MedInc` (renda) varia de 0,5 a 15, enquanto `Population` varia de 3 a 35.000. Essa diferença de 1000x é o exato motivo pelo qual precisamos padronizar os dados.

**3. Boxplot: Anatomia Matemática e Detecção de Outliers**
O Boxplot é um gráfico que resume uma distribuição numérica em 5 elementos visuais, baseados nos **quartis**:
* A **linha inferior da caixa** marca o 1º quartil (Q1 = 25%).
* A **linha central da caixa** marca a mediana (Q2 = 50%).
* A **linha superior da caixa** marca o 3º quartil (Q3 = 75%).
* A altura da caixa é o **IQR (Intervalo Interquartil):** `IQR = Q3 - Q1`.
* Os "bigodes" (*whiskers*) se estendem até 1.5 × IQR acima de Q3 e abaixo de Q1. 
* Qualquer ponto plotado **além** dos bigodes é matematicamente classificado como um **outlier** (valor atípico).

No nosso projeto, o Boxplot da coluna `MedInc` (renda mediana) no California Housing revelou a presença de outliers na faixa superior, ou seja, bairros com renda mediana muito acima da norma geral. Essa informação é crucial porque modelos sensíveis a escala (como Regressão Linear, SVM e Redes Neurais) terão seus coeficientes distorcidos por esses pontos extremos, enquanto modelos baseados em árvores (Random Forest, XGBoost) são naturalmente imunes a outliers, pois tomam decisões baseadas em limiares de corte (splits), não em distâncias geométricas.

**4. Histograma, Distribuição e Assimetria (*Skewness*)**
O Histograma divide o intervalo de valores de uma variável em faixas ("bins") e conta quantas observações caem em cada faixa, formando barras verticais. Diferente de um gráfico de barras simples (que compara categorias distintas como "Masculino" e "Feminino"), o Histograma representa uma **distribuição contínua** de probabilidade.

No histograma do preço dos imóveis (`MedHouseVal`), observamos uma **assimetria à direita** (*right-skewed* ou *positive skew*): a maioria das casas se concentra em preços baixos/médios, mas existe uma cauda longa de casas muito caras puxando a distribuição para a direita. Em distribuições assimétricas, a média é arrastada na direção da cauda (ficando maior que a mediana), o que pode enganar análises ingênuas. Essa assimetria no California Housing também explica por que o histograma mostra um "pico" artificial no valor máximo (~5.0): o dataset original limitou (*capped*) todos os preços acima de $500K a esse teto.

**5. Desbalanceamento de Classes**
No dataset Breast Cancer, a verificação revelou que 62.7% das amostras são tumores benignos e 37.3% são malignos. Isso configura um **desbalanceamento moderado**. Em cenários extremos (ex: 99% benigno, 1% maligno), um modelo "preguiçoso" que sempre preveja "benigno" atingiria 99% de acurácia sem ter aprendido absolutamente nada sobre tumores malignos. Por isso, a **Acurácia** sozinha é uma métrica enganosa em datasets desbalanceados.

No contexto médico, um **Falso Negativo** (classificar um tumor maligno como benigno) é catastroficamente mais grave do que um Falso Positivo (classificar um benigno como maligno, gerando apenas um susto). É por isso que, no Dia 5, quando avaliarmos os modelos de classificação, daremos atenção especial ao **Recall** (Sensibilidade): a taxa de acerto especificamente entre os tumores que são de fato malignos. O **F1-Score**, que é a média harmônica entre Precisão e Recall, também será crucial como métrica balanceada.

**6. Divisão Treino/Teste: O Método Holdout**
A divisão Treino/Teste é o mecanismo fundamental para estimar o desempenho real de um modelo em dados que ele nunca viu. A analogia é direta: o professor (treino) ensina a matéria, e a prova (teste) mede se o aluno aprendeu ou apenas decorou.

Aplicamos uma proporção de **80/20** (80% treino, 20% teste), que é a convenção mais consolidada na literatura. Para o Breast Cancer, isso resultou em **455 amostras de treino** e **114 amostras de teste**; para o California Housing, **16.512 de treino** e **4.128 de teste**.

Existe um conceito chamado **Overfitting** (Sobreajuste): ocorre quando o modelo se ajusta excessivamente aos dados de treino, memorizando os padrões específicos (incluindo o ruído) daquele conjunto em vez de aprender regras generalizáveis. Um modelo em overfitting apresenta desempenho excelente no treino mas péssimo no teste. A separação Treino/Teste é a primeira linha de defesa contra essa falha.

**7. Estratificação (*Stratified Sampling*)**
No `train_test_split` do dataset de classificação (Breast Cancer), usamos o parâmetro `stratify=y_clf`. Esse parâmetro força a função a manter exatamente a mesma proporção de classes (62/38) tanto no conjunto de treino quanto no de teste. Sem estratificação, a divisão aleatória poderia, por puro azar estatístico, colocar quase todos os tumores malignos no treino e quase nenhum no teste, gerando métricas de teste completamente não-representativas.

Não usamos `stratify` no California Housing porque a variável alvo é contínua (um preço numérico). Não existem "classes" discretas para serem proporcionalmente distribuídas. Tentar estratificar por uma variável contínua geraria um erro de código, pois cada preço é praticamente único.

**8. StandardScaler: A Fórmula do Z-Score**
O `StandardScaler` aplica, para cada coluna, a transformação Z-Score:

`z = (x - μ) / σ`

Onde `x` é o valor original, `μ` é a média da coluna e `σ` é o desvio padrão. Após a transformação, cada coluna terá média = 0 e desvio padrão = 1. Essa padronização é **obrigatória** para algoritmos que dependem do cálculo de distância geométrica:
* **K-Means e DBSCAN** calculam distância Euclidiana entre pontos. Se uma coluna varia de 0 a 35.000 e outra de 0 a 15, a primeira dominará totalmente o cálculo de distância.
* **SVM** (Support Vector Machine) encontra o hiperplano que maximiza a margem entre classes. Features com escalas maiores distorcem a orientação do hiperplano.
* **Redes Neurais (MLP)** usam Gradiente Descendente para ajustar seus pesos. Features em escalas muito diferentes fazem com que o gradiente oscile violentamente em algumas direções e avance lentamente em outras, prejudicando a convergência.

Modelos baseados em **árvores de decisão** (Decision Tree, Random Forest, Gradient Boosting, XGBoost) são naturalmente imunes a diferenças de escala, pois tomam decisões baseadas em limiares de corte ordenados ("a renda é > 5.0?"), não em distâncias geométricas. Ainda assim, padronizamos todos os dados para manter um pipeline único e consistente.

**9. Data Leakage (Vazamento de Dados)**
O `scaler.fit()` (que calcula a média e o desvio padrão) deve ser executado **exclusivamente** no conjunto de treino (`X_train`), e o `scaler.transform()` é então aplicado tanto no treino quanto no teste usando os parâmetros aprendidos no treino.

Se rodássemos `fit()` na base inteira (treino + teste), a média e o desvio padrão conteriam informações estatísticas dos dados de teste. Isso significa que o modelo estaria, indiretamente, "olhando a prova antes de fazê-la". Esse fenômeno é chamado de **Data Leakage** (Vazamento de Dados) e é um dos erros mais graves (e mais silenciosos) em projetos de Machine Learning, pois infla artificialmente as métricas de desempenho no teste sem que o praticante perceba.

**10. Alternativa: MinMaxScaler**
Além do `StandardScaler`, existe o **MinMaxScaler**, que transforma os dados para o intervalo [0, 1] usando a fórmula:

`x_scaled = (x - x_min) / (x_max - x_min)`

O MinMaxScaler é preferível em cenários onde: (a) você sabe que os dados não possuem outliers severos (pois um único outlier extremo comprime todos os outros valores perto de zero); (b) o algoritmo exige entradas no intervalo [0, 1] (como redes neurais com funções de ativação sigmoide). Para o nosso projeto, escolhemos o StandardScaler porque ele é mais robusto a outliers (que confirmamos existir no Boxplot) e é a recomendação padrão da documentação do `scikit-learn` para pipelines de classificação e regressão.

### ❓ 10 Perguntas da Banca sobre a Análise Exploratória (EDA)
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

### ❓ 10 Perguntas da Banca sobre o Pré-Processamento
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

---

# 📅 DIA 3: Modelagem Não Supervisionada e Agrupamentos

Neste dia, aplicamos métodos que operam na ausência de rótulos (gabaritos). O objetivo é encontrar padrões estruturais, projetar dados geometricamente e tentar agrupar instâncias de forma autônoma.

### 📚 Fundamentação Teórica do Dia 3

**1. PCA (Análise de Componentes Principais)**
O PCA é uma técnica de álgebra linear (redução de dimensionalidade) que cria novos eixos ortogonais, chamados de Componentes Principais, que maximizam a variância retida dos dados originais. Ao invés de plotar as 30 dimensões do Breast Cancer, o PCA encontrou os *Autovetores (Eigenvectors)* da matriz de covariância. O gráfico de "Variância Acumulada" prova quantos componentes precisamos para reter pelo menos 90% da informação. A projeção 2D permite ao cérebro humano enxergar a separação biológica dos tumores.

**2. K-Means e Método do Cotovelo (*Elbow Method*)**
O K-Means é um algoritmo baseado em centroides que tenta minimizar a "Inércia" (soma dos erros quadráticos, WCSS). 
* **k-means++**: Em vez de chutar posições aleatórias para os centroides e correr o risco de cair num mínimo local ruim, o inicializador `k-means++` os espalha de forma estatisticamente inteligente na primeira iteração.
* **Elbow Method**: A Inércia sempre cai quando aumentamos o $K$ (número de clusters). O cotovelo mostra o ponto de equilíbrio onde adicionar mais clusters deixa de trazer ganhos matemáticos significativos.

**3. DBSCAN (Agrupamento Espacial Baseado em Densidade)**
Ao contrário do K-Means, o DBSCAN não precisa que você diga quantos clusters existem. Ele agrupa pontos contínuos que possuem alta densidade.
* Parâmetro $\varepsilon$ (Eps): O raio geográfico em torno de um ponto.
* Parâmetro *minPts*: O mínimo de pontos que devem existir dentro desse raio para ele ser considerado um cluster válido.
O DBSCAN é brilhante para capturar formas geométricas arbitrárias (que o K-Means erraria por assumir que tudo é redondo) e, mais importante, ele joga dados isolados para um grupo chamado "Ruído" (Noise / -1), ajudando a descobrir anomalias.

**4. Métricas de Avaliação**
* **Métricas Internas (Sem gabarito):** Inércia (coesão), Silhouette (mede se um ponto está mais perto do seu próprio cluster do que do cluster vizinho), Calinski-Harabasz (dispersão entre grupos vs intra-grupos) e Davies-Bouldin (menor significa clusters mais separados).
* **Métricas Externas (Com gabarito):** ARI (Adjusted Rand Index) e NMI (Normalized Mutual Information). Aplicadas exclusivamente no Breast Cancer (pois temos as classes originais para comparar), avaliam se os clusters que o algoritmo encontrou sozinhos batem com a biologia real (Maligno/Benigno). Não usamos no California Housing pois seu alvo é contínuo (R$).

### ❓ 10 Perguntas da Banca sobre Modelagem Não Supervisionada
1. O que o PCA faz matematicamente? Ele exclui colunas irrelevantes ou ele as combina? *(Resposta: Ele não exclui, ele cria combinações lineares ortogonais de todas as colunas originais).*
2. No gráfico de variância acumulada, por que nós traçamos uma linha vermelha em 90%?
3. O algoritmo K-Means sofre muito com inicializações aleatórias (mínimos locais). Como o nosso uso do `init='k-means++'` no código evitou isso?
4. A Inércia mede o erro quadrático dentro do cluster. Por que a Inércia não serve como métrica para o DBSCAN?
5. Como vocês justificam o número de clusters (K) escolhido para o dataset da Califórnia usando a curva do Método do Cotovelo?
6. O K-Means assume que os clusters têm formato esférico. Se os dados tivessem o formato de "duas luas entrelaçadas", qual algoritmo venceria: K-Means ou DBSCAN? Por quê?
7. Como vocês calibraram os parâmetros Epsilon ($\varepsilon$) e *Min_samples* no DBSCAN? Qual o impacto se o $\varepsilon$ fosse alto demais? *(R: Tudo viraria um único cluster gigantesco).*
8. O Silhouette Score vai de -1 a 1. O que significa matematicamente se um ponto receber o score de -0.9? *(R: Que ele foi agrupado no cluster errado e está muito mais próximo do cluster vizinho).*
9. Por que não foi possível calcular as métricas ARI e NMI para o dataset California Housing, mas sim para o Breast Cancer?
10. Se a redução de dimensionalidade por PCA funciona tão bem para o Breast Cancer, poderíamos treinar os modelos de classificação dos próximos dias usando apenas os 2 componentes principais em vez das 30 colunas originais? O que ganharíamos e o que perderíamos?

---

# 📅 DIA 4: Modelagem Supervisionada — Regressão

Neste dia, colocamos os dados padronizados do California Housing para serem consumidos por 6 algoritmos distintos de regressão. O objetivo é treinar cada modelo, medir seu desempenho no conjunto de teste (dados que ele nunca viu) e comparar os resultados em uma tabela unificada para identificar qual algoritmo generaliza melhor para o problema de predição de preços imobiliários.

### 📚 Fundamentação Teórica do Dia 4

**1. O Problema de Regressão na Perspectiva Matemática**
Em um problema de regressão, dado um vetor de entrada $X$ (os atributos de um bairro: renda, idade do imóvel, latitude, longitude...) e um valor alvo $y$ (o preço mediano das casas), o objetivo do modelo é aprender uma função $f$ tal que $\hat{y} = f(X)$ minimize a diferença entre o valor previsto $\hat{y}$ e o valor real $y$. Cada um dos 6 algoritmos que treinamos usa uma estratégia matemática radicalmente diferente para construir essa função $f$, e é exatamente por isso que seus desempenhos variam.

**2. Os 6 Algoritmos de Regressão**

**2.1. Regressão Linear (*Linear Regression*)**
É o algoritmo mais antigo e mais simples de regressão. Ele assume que a relação entre as features e o preço é **linear** (uma reta em 2D, um hiperplano em múltiplas dimensões). A fórmula geral é:

`ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`

Onde cada $β$ (beta) é um coeficiente de peso que o modelo calcula usando o método dos **Mínimos Quadrados Ordinários (OLS)**: ele encontra os valores de $β$ que minimizam a soma dos quadrados dos resíduos (a diferença entre o preço real e o previsto, elevada ao quadrado). A Regressão Linear é extremamente rápida (0,08 segundos no nosso teste), mas sua limitação fundamental é a **linearidade**: ela não consegue capturar relações curvas ou interações complexas entre variáveis. No nosso experimento, ela obteve o pior $R^2$ (0.575), confirmando que a relação entre preço e geografia da Califórnia é intrinsecamente não-linear.

**2.2. Árvore de Decisão (*Decision Tree Regressor*)**
A Árvore de Decisão divide recursivamente o espaço de features em regiões retangulares, fazendo perguntas binárias do tipo "a renda mediana é > 5.0?". Em cada "folha" da árvore, a predição é a **média** dos preços de todas as casas que caíram naquela região. A grande vantagem é que ela captura relações **não-lineares** naturalmente, sem precisar que o engenheiro especifique a forma matemática da relação. Porém, a Árvore de Decisão sofre de **alta variância**: ela tende a memorizar os dados de treino (Overfitting), criando ramificações excessivamente específicas. No nosso teste, ela obteve $R^2 = 0.623$, melhor que a Regressão Linear mas bem abaixo dos modelos ensemble.

**2.3. Random Forest Regressor (Floresta Aleatória)**
O Random Forest é uma técnica de **Ensemble Learning** (Aprendizado por Comitê): ele treina centenas de Árvores de Decisão (no scikit-learn, 100 por padrão), cada uma com uma amostra diferente dos dados (técnica chamada **Bagging**, ou *Bootstrap Aggregating*) e uma seleção aleatória de features. A predição final é a **média** das predições de todas as árvores. Essa estratégia de "sabedoria das multidões" reduz drasticamente a variância (Overfitting) que afligia a árvore individual. No nosso teste, o Random Forest alcançou $R^2 = 0.804$, um salto enorme em relação à árvore individual (0.623). Usamos `n_jobs=-1` para distribuir o treinamento das 100 árvores pelos 4 núcleos da CPU simultaneamente (paralelismo).

**2.4. XGBoost Regressor (*Extreme Gradient Boosting*)**
O XGBoost é outro método de Ensemble, mas sua filosofia é oposta à do Random Forest. Em vez de treinar árvores **independentes** e tirá-las a média (Bagging), o XGBoost treina árvores de forma **sequencial**: cada nova árvore é treinada para corrigir exclusivamente os erros (resíduos) que a árvore anterior cometeu. Esse processo iterativo de correção é chamado de **Boosting** (impulsionamento). Além disso, o XGBoost incorpora **regularização** ($L_1$ e $L_2$) nos pesos das árvores para controlar o Overfitting, e seu código interno é altamente otimizado em C++ com cache-awareness e paralelismo nativo. No nosso teste, ele foi o **campeão absoluto**: $R^2 = 0.836$, o maior poder preditivo, com um tempo de treinamento de apenas 1,18 segundos. Isso explica por que o XGBoost domina competições de Ciência de Dados no Kaggle há anos.

**2.5. SVR (*Support Vector Regression*)**
O SVR é a versão de regressão do SVM (Support Vector Machine). Enquanto o SVM de classificação busca o hiperplano que maximiza a margem de separação entre classes, o SVR busca uma "faixa" (tubo de largura $\varepsilon$, chamada *epsilon-insensitive tube*) que contenha o máximo possível de pontos de dados. Os pontos que ficam fora do tubo são chamados de **vetores de suporte** e são os únicos que contribuem para a função de perda. O SVR pode capturar relações não-lineares usando o **Kernel Trick** (por padrão, o kernel RBF — *Radial Basis Function*), que projeta os dados para um espaço de dimensão superior onde uma separação linear se torna possível. No nosso teste, ele obteve $R^2 = 0.727$, um resultado intermediário. Entretanto, seu tempo de treinamento foi **26,22 segundos** — o segundo mais lento — porque a complexidade computacional do SVR escala proporcionalmente a $O(n^2)$ ou $O(n^3)$ com o número de amostras, tornando-o extremamente lento para datasets grandes como o California Housing (20.640 linhas).

**2.6. MLP Regressor (Rede Neural Artificial / *Multi-Layer Perceptron*)**
A MLP é uma rede neural composta por camadas de **neurônios artificiais** (também chamados de perceptrons). Configuramos duas camadas ocultas com 100 e 50 neurônios, respectivamente (`hidden_layer_sizes=(100, 50)`). Cada neurônio recebe entradas ponderadas, soma-as, aplica uma **função de ativação** não-linear (por padrão, a ReLU — *Rectified Linear Unit*: $f(x) = \max(0, x)$), e propaga o resultado para a próxima camada. A camada de saída possui um único neurônio com **ativação linear** (identidade), pois precisamos prever um número contínuo, não uma classe.

O treinamento ocorre pelo algoritmo **Backpropagation** combinado com um otimizador (por padrão, o Adam). A cada passagem pelos dados (época), a rede calcula o erro (MSE), propaga esse erro de volta pelas camadas e ajusta os pesos de cada conexão na direção que reduz o erro (Gradiente Descendente). Configuramos `max_iter=500` para dar à rede tempo suficiente de convergir. No nosso teste, a MLP obteve $R^2 = 0.793$, empatando tecnicamente com o Random Forest, mas ao custo de **74,94 segundos** de treinamento — o mais lento de todos. Isso ocorre porque ela precisa iterar centenas de vezes sobre os dados, ajustando milhares de pesos sinápticos.

**3. As 4 Métricas Obrigatórias de Regressão**

**3.1. MAE — Mean Absolute Error (Erro Médio Absoluto)**

`MAE = (1/n) * Σ |yᵢ - ŷᵢ|`

Calcula a média das diferenças absolutas entre o valor real e o previsto. É a métrica mais intuitiva: "em média, o modelo erra por X unidades". No nosso caso, o XGBoost errou por 0.31 (31 mil dólares em média). O MAE trata todos os erros de forma igual, independentemente de serem grandes ou pequenos.

**3.2. MSE — Mean Squared Error (Erro Quadrático Médio)**

`MSE = (1/n) * Σ (yᵢ - ŷᵢ)²`

Eleva os erros ao quadrado antes de calcular a média. Isso **penaliza desproporcionalmente os erros grandes**: um erro de 10 contribui com 100, mas um erro de 100 contribui com 10.000. O MSE é a função de perda padrão usada internamente pela Regressão Linear (OLS) e pela MLP (Backpropagation). Sua desvantagem é que a unidade fica elevada ao quadrado (ex: "dólares²"), dificultando a interpretação direta.

**3.3. RMSE — Root Mean Squared Error (Raiz do Erro Quadrático Médio)**

`RMSE = √MSE`

É simplesmente a raiz quadrada do MSE. A vantagem é que o RMSE retorna à **unidade original** da variável alvo (dólares), tornando-o interpretável. Ele mantém a propriedade de penalizar erros grandes (herdada do MSE). No nosso teste, o RMSE do XGBoost foi 0.462 (~46 mil dólares), enquanto o da Regressão Linear foi 0.745 (~74 mil dólares).

**3.4. R² — Coeficiente de Determinação**

`R² = 1 - (Σ (yᵢ - ŷᵢ)²) / (Σ (yᵢ - ȳ)²)`

O $R^2$ compara o erro do modelo com o erro de um "modelo burro" que sempre prevê a média ($\bar{y}$). Seu valor varia de $-\infty$ a $1$:
* $R^2 = 1$: Previsão perfeita. O modelo acertou todos os preços exatamente.
* $R^2 = 0$: O modelo é tão ruim quanto simplesmente chutar a média geral dos preços.
* $R^2 < 0$: O modelo é **pior** do que chutar a média (isso acontece em modelos severamente inadequados).

O $R^2$ é a métrica mais poderosa para comparar modelos de regressão porque é **adimensional** (não depende da escala dos preços) e porque tem uma interpretação direta: "o modelo explica X% da variação nos preços". O XGBoost explicou 83.6% da variação, enquanto a Regressão Linear explicou apenas 57.5%.

**4. Análise Crítica dos Resultados Experimentais**
Os resultados do nosso experimento confirmam três padrões teóricos fundamentais da literatura de Machine Learning:

* **Modelos Ensemble superam modelos individuais:** Random Forest ($R^2 = 0.804$) e XGBoost ($R^2 = 0.836$) esmagaram a Decision Tree individual ($R^2 = 0.623$). Isso valida experimentalmente que combinar múltiplos modelos fracos produz um modelo forte (o princípio do *Ensemble Learning*).

* **Boosting supera Bagging:** O XGBoost (Boosting sequencial) superou o Random Forest (Bagging paralelo) por 3 pontos percentuais de $R^2$. Isso é consistente com a teoria: o Boosting foca cirurgicamente nos erros residuais, enquanto o Bagging trata todas as amostras com igual importância.

* **Complexidade computacional vs. ganho:** A MLP consumiu 74,94 segundos (63x mais que o XGBoost) para atingir um $R^2$ inferior. Para datasets tabulares estruturados (tabelas de números), modelos baseados em árvores (XGBoost, Random Forest) consistentemente superam Redes Neurais. As Redes Neurais brilham em dados não-estruturados (imagens, áudio, texto), não em tabelas de CSV.

* **A Regressão Linear como baseline:** Apesar de ser o pior modelo, a Regressão Linear cumpre um papel vital: ela serve como **baseline** (referência mínima). Se nenhum dos modelos complexos conseguisse superar a Regressão Linear, isso indicaria que o problema não tem relações preditivas nos dados ou que há um erro grave no pré-processamento.

### ❓ 10 Perguntas da Banca sobre Modelagem de Regressão
1. Por que a Regressão Linear foi o pior modelo no California Housing? Qual premissa matemática ela viola nesse dataset?
2. Qual é a diferença conceitual entre **Bagging** (Random Forest) e **Boosting** (XGBoost)? Por que o Boosting tendeu a superar o Bagging neste experimento?
3. A Decision Tree obteve $R^2 = 0.623$ enquanto o Random Forest obteve $R^2 = 0.804$. O que exatamente o Random Forest faz de diferente para obter esse ganho tão significativo?
4. O que significa, na prática, dizer que o XGBoost possui $R^2 = 0.836$? Ele está explicando o quê?
5. Qual a diferença entre MAE e RMSE? Em que cenário um engenheiro deveria preferir o MAE ao RMSE? *(Resposta: Quando outliers são esperados e não devem ser penalizados desproporcionalmente).*
6. O SVR demorou 26 segundos para treinar. Por que a complexidade computacional do SVM escala tão mal com o número de amostras ($O(n^2)$ a $O(n^3)$)?
7. A MLP foi configurada com `hidden_layer_sizes=(100, 50)`. O que aconteceria se usássemos apenas `(10,)` (uma única camada oculta com 10 neurônios)?
8. Por que configuramos `max_iter=500` na MLP? O que aconteceria se deixássemos o padrão de 200? *(Resposta: A rede poderia não convergir e exibir um ConvergenceWarning, com métricas piores).*
9. Vocês usaram `n_jobs=-1` no Random Forest e no XGBoost. O que esse parâmetro faz e qual é o ganho real? Ele funcionaria no SVR?
10. Se o $R^2$ de um modelo fosse **negativo**, o que isso significaria? Seria possível? *(Resposta: Sim, significaria que o modelo é pior do que simplesmente prever a média dos preços para todas as casas).*
