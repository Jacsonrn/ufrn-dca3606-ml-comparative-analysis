# 🤖 Análise Comparativa de Algoritmos de Machine Learning

Projeto desenvolvido para a disciplina **DCA3606 - Inteligência Artificial** na Universidade Federal do Rio Grande do Norte (UFRN).

## 🎯 Visão Geral
Este repositório contém uma análise comparativa rigorosa de algoritmos de Machine Learning (Supervisionados e Não Supervisionados). Foram avaliados 12 modelos supervisionados e 2 não supervisionados utilizando rigor metodológico, métricas acadêmicas e análise crítica baseada em conceitos como Trade-off Viés-Variância, Teorema *No Free Lunch* e a Navalha de Occam.

## 📂 Datasets Utilizados
1. **Classificação:** [Breast Cancer Wisconsin](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (Diagnóstico tumoral biológico)
2. **Regressão:** [California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (Precificação imobiliária espacial)

## 🏆 Resultados Principais (Top 1)
- **Regressão (California Housing):** O **XGBoost** foi o campeão absoluto com **R² = 0.836**, superando modelos robustos como Random Forest e Redes Neurais (MLP). Seu processo de *Gradient Boosting* foi cirúrgico na correção sequencial de resíduos espaciais não-lineares.
- **Classificação (Breast Cancer):** A **Regressão Logística** e o **SVC** empataram no topo com **F1-Score = 0.986**. A projeção bidimensional do PCA comprovou que os dados eram altamente separáveis linearmente, fazendo com que a simplicidade paramétrica vencesse a complexidade dos Ensembles, evitando Overfitting.
- **Agrupamento:** O **K-Means** demonstrou melhor formação de clusters que o DBSCAN. O DBSCAN falhou nos dados da Califórnia devido à densidade contínua do mapa geográfico, que não possuía "vales de separação" naturais.

## 📁 Estrutura do Repositório
- 📂 `notebooks/` 
  - `01_eda_e_preprocessamento.ipynb` — Análise Exploratória (EDA) e Escalonamento.
  - `02_modelagem_nao_supervisionada.ipynb` — PCA, K-Means e DBSCAN.
  - `03_modelagem_supervisionada_regressao.ipynb` — 6 modelos preditivos aplicados a habitação.
  - `04_modelagem_supervisionada_classificacao.ipynb` — 6 modelos de diagnóstico médico.
  - `05_relatorio_final_consolidado.ipynb` — Tabelas de comparação e análise crítica profunda.
- 📄 `Fundamentacao_Teorica.md` — Documentação acadêmica exaustiva justificando toda a matemática dos algoritmos, ideal para consultas e defesas orais.
- 📊 `Apresentacao_Projeto_ML.pptx` — Apresentação oficial do projeto (17 slides).

## 🚀 Como Executar Localmente

```bash
# Clone o repositório
git clone https://github.com/Jacsonrn/ufrn-dca3606-ml-comparative-analysis.git
cd ufrn-dca3606-ml-comparative-analysis

# Crie e ative o ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

## 👨‍💻 Autor
**Jacson Arruda Ribeiro**  
Universidade Federal do Rio Grande do Norte (UFRN) — Instituto Metrópole Digital (IMD / nPITI)
