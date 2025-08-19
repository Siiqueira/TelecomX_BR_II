# 🛰️ Projeto: Previsão de Cancelamento de Clientes (Churn) - TelecomX_BR

## 🎯 Missão do Cientista de Dados
Desenvolver um pipeline de machine learning completo para prever a evasão (churn) de clientes da TelecomX_BR, permitindo ações proativas de retenção por parte da empresa.

---

## 📌 Objetivos do Desafio

- Preparar os dados para modelagem (tratamento, encoding, balanceamento);
- Analisar a correlação entre variáveis e selecionar as mais relevantes;
- Treinar e comparar diferentes modelos preditivos;
- Avaliar o desempenho com métricas robustas;
- Interpretar resultados com foco estratégico em retenção;
- Gerar relatório técnico completo para stakeholders;
- Salvar modelo final com pipeline e encoding embutidos.

---

## 📁 Estrutura do Projeto

```bash
📦 projeto_churn/
├── data/
│   ├── raw/                         # Dados brutos
│   ├── processed/                   # Dados tratados
│   └── results/                     # CSVs e gráficos gerados
├── models/
│   └── modelo_final.pkl             # Modelo final salvo
├── notebooks/
│   └── modelo_churn.ipynb          # Notebook principal do projeto
├── src/
│   └── pipeline/                    # Funções para o pipeline e modelagem
├── README.md
├── Relatorio_Tecnico.md
└── Arquitetura_Projeto.md
```


## ⚙️ Tecnologias e Bibliotecas Utilizadas

| Categoria              | Bibliotecas                                                     |
|-----------------------|----------------------------------------------------------------|
| Manipulação de dados   | pandas, numpy                                                  |
| Visualização          | matplotlib, seaborn                                            |
| Modelagem             | scikit-learn, imbalanced-learn                                 |
| Avaliação de modelos  | classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, learning_curve |
| Pipeline e Hiperparâmetros | Pipeline, RandomizedSearchCV, GridSearchCV                |
| Serialização          | pickle                                                        |
| Outros                | warnings, StratifiedKFold                                      |

---

## 📈 Modelo Final

| Item                     | Descrição                                                   |
|--------------------------|-------------------------------------------------------------|
| Modelo escolhido         | Random Forest Classifier                                    |
| Técnica de balanceamento  | Oversampling (SMOTE)                                        |
| Top Features             | tempo_contrato_Mensal, permanencia, fibra_optica, total_gastos, pagamento_Cheque_Digital |
| Accuracy final           | 0.77                                                        |
| Recall (classe 1 - churn)| 0.78                                                        |
| F1-score (classe 1)      | 0.62                                                        |
| ROC AUC                  | 0.82                                                        |

---

## 🧪 Teste com Dados de Validação

| Classe | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| 0      | 0.93      | 0.81   | 0.87     | 16      |
| 1      | 0.57      | 0.80   | 0.67     | 5       |
| **Acurácia** |       |        | **0.81** | **21**  |

⚠️ O conjunto de validação possui apenas 21 amostras, portanto os resultados devem ser interpretados com cautela.

---

## 🧠 Conclusão Estratégica

Clientes com contratos mensais, baixa permanência, e que utilizam fibra óptica representam o maior risco de churn.

Estratégias de retenção ativa devem focar nesses perfis.

O modelo, mesmo diante de dados desbalanceados, apresenta capacidade robusta de generalização e é adequado para uso em produção.

---

## 📌 Observações

Este projeto é voltado para análise de churn com dados sintéticos e objetivos didáticos.

Para implementação em produção, recomenda-se:

- Monitoramento contínuo de métricas;
- Coleta de novas variáveis comportamentais;
- Atualização periódica do modelo.

---

## 📬 Contato

Para dúvidas, sugestões ou colaborações, entre em contato com o time de ciência de dados da TelecomX_BR.

---

## ✅ Modelos Testados

| Modelo                    | AUC     | Recall Classe 1 | F1 Classe 1 | Accuracy |
|--------------------------|---------|------------------|-------------|----------|
| Decision Tree            | 0.73    | 0.69             | 0.57        | 0.72     |
| Random Forest            | 0.82    | 0.71             | 0.61        | 0.76     |
| Random Forest + SMOTE    | 0.82    | 0.78             | 0.62        | 0.74     |

---

## ⭐ Modelo Final Escolhido: `RandomForestClassifier` com Oversampling (SMOTE)

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
## Tuning com RandomizedSearchCV

- Melhor desempenho em recall da classe positiva (cancelamento)
- Selecionadas as 5 features mais relevantes

### 📊 Desempenho Final

**Teste com Dados de Validação**

| Classe | Precision | Recall | F1-score | Suporte |
|--------|-----------|--------|----------|---------|
| 0      | 0.93      | 0.81   | 0.87     | 16      |
| 1      | 0.57      | 0.80   | 0.67     | 5       |

- **Acurácia Geral:** 81%  
- **Weighted F1-score:** 0.82

### 🗂️ Links Importantes

- 📈 Gráficos de correlação e distribuição  
- 📊 Matriz de confusão  
- 📊 Curva ROC AUC  
- 📊 Curva Precisão x Recall  
- 📊 Features Importances (CSV + gráfico)  
- 📊 Resultados de validação cruzada  
- 📊 Comparações entre modelos  


📦 Como Usar

Clone o repositório:

git clone https://github.com/seuusuario/telecom_churn_prediction.git


Instale as dependências:

pip install -r requirements.txt


Execute os notebooks disponíveis na pasta notebooks/.

Avalie o modelo final ou carregue o modelo .pkl com o seguinte código:

import pickle
with open('models/modelo_final.pkl', 'rb') as f:
    model = pickle.load(f)

🧠 Conclusão

Este projeto entregou uma solução preditiva eficiente para o problema de churn, com foco em clientes com maior risco de cancelamento, variáveis mais influentes e estratégias práticas de retenção. O modelo Random Forest com SMOTE demonstrou robustez, generalização e capacidade de auxiliar a empresa na tomada de decisões estratégicas.

📮 Contato

👨‍💻 Desenvolvido por: [Seu Nome]
📧 Email: seu.email@dominio.com
🔗 LinkedIn: linkedin.com/in/seu-perfil

--- 
Relatorio_Tecnico.md

# 📊 Relatório Técnico – Análise do Modelo de Previsão de Evasão de Clientes (Churn)

---

## 📌 Sumário

1. [Carregamento e Tratamento dos Dados](#1-carregamento-e-tratamento-dos-dados)
2. [Análise Exploratória e Correlações](#2-análise-exploratória-e-correlações)
3. [Modelagem Inicial](#3-modelagem-inicial)
4. [Avaliação dos Modelos](#4-avaliação-dos-modelos)
5. [Balanceamento dos Dados](#5-balanceamento-dos-dados)
6. [Modelo Final – Ajuste de Hiperparâmetros](#6-modelo-final--ajuste-de-hiperparâmetros)
7. [Teste com Dados de Validação](#7-teste-com-dados-de-validação)
8. [Conclusão Estratégica](#8-conclusão-estratégica)

---

## 1. Carregamento e Tratamento dos Dados

Os dados foram carregados e tratados com as seguintes ações:

- Remoção de colunas irrelevantes
- Transformação de variáveis categóricas com One Hot Encoding
- Análise de proporção da variável alvo (target)

### Proporção do Target (Cancelamento):

- **Não Cancelaram (classe 0):** 73%
- **Cancelaram (classe 1):** 27%

🔍 *Os dados estão desbalanceados. Isso pode causar viés nos modelos preditivos. Técnicas de oversampling e undersampling foram testadas para mitigar esse problema.*

---

## 2. Análise Exploratória e Correlações

### Principais descobertas:

- **Gastos Mensais** e **Gastos Diários** têm **correlação perfeita**. Decidimos remover `gastos_mensais`.
- Contrato Mensal tem forte correlação com cancelamento (**+40%**).
- Clientes com maior permanência têm menor chance de cancelar (**correlação negativa de -35%**).
- Fibra Óptica está correlacionada com gastos elevados e maior taxa de churn.
- Clientes com contratos mais curtos tendem a acumular menos gastos e cancelar mais cedo.

📈 **Links para gráficos:**
- [🔗 Gráfico de Correlação](#coloque-aqui-seu-link)
- [🔗 Tempo de Contrato x Cancelamento](#coloque-aqui-seu-link)
- [🔗 Total de Gasto x Cancelamento](#coloque-aqui-seu-link)

---

## 3. Modelagem Inicial

### Separação dos dados:

- **Dados de treino:** (5608, 26)
- **Dados de teste:** (1403, 26)

### Modelos testados:

1. **Decision Tree Classifier**
```python
DecisionTreeClassifier(
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
## 4. Avaliação dos Modelos

### 🔢 Métricas Comparativas

| Modelo         | AUC  | Recall (Classe 1) | F1 (Classe 1) | Accuracy |
|----------------|------|-------------------|---------------|----------|
| Decision Tree  | 0.73 | 0.69              | 0.57          | 0.72     |
| Random Forest  | 0.82 | 0.71              | 0.61          | 0.76     |

### 📉 Matriz de Confusão

🔗 [Link para o gráfico]

> Random Forest foi superior em quase todas as métricas, principalmente na identificação correta de clientes que não cancelam (classe 0).  
> A classe 1 (cancelamento) ainda apresenta desafios.

### 📈 Curva ROC AUC

🔗 [Link para o gráfico]

- Decision Tree: AUC = 0.73  
- Random Forest: AUC = 0.82

### 📈 Curva Precisão x Recall

🔗 [Link para o gráfico]

- Decision Tree: Avg. Precision = 0.44  
- Random Forest: Avg. Precision = 0.59

### 📄 Classification Report

🔗 [Link para CSV com os relatórios]

> Random Forest teve melhor desempenho para a classe de interesse (churn).  
> F1-Score e recall da classe 1 foram superiores ao Decision Tree.

### 🔁 Validação Cruzada

🔗 [Link para CSV]

> Resultados mais estáveis e consistentes com Random Forest.  
> Menor variância e melhor generalização.

---

## 5. Balanceamento dos Dados

**Técnicas utilizadas:**

- Oversampling (SMOTE) ✅ escolhido  
- Undersampling (NearMiss)

📄 🔗 [CSV com resultados]

**Escolha: Oversampling com SMOTE**  
✅ Apresentou melhor equilíbrio entre precisão e recall, além de maior acurácia (0.79).

---

## 6. Modelo Final – Ajuste de Hiperparâmetros

### 🔧 RandomizedSearchCV

| precision | recall | f1-score | support |
|-----------|--------|----------|---------|
| 0         | 0.88   | 0.79     | 0.83    | 1030    |
| 1         | 0.55   | 0.71     | 0.62    | 373     |
| **accuracy** |        |          | 0.77    | 1403    |

> Modelo robusto, mas com trade-off entre custo de retenção (precisão baixa) e eficácia na detecção (recall alto).

### 🔍 GridSearchCV

> Resultados semelhantes ao RandomizedSearchCV.  
> GridSearch foi descartado por não apresentar ganho relevante e ter maior tempo de execução.

### 🔝 Seleção das Melhores Features

| Classe | Precision | Recall | F1-score | Suporte |
|--------|-----------|--------|----------|---------|
| 0      | 0.90      | 0.73   | 0.81     | 1030    |
| 1      | 0.51      | 0.78   | 0.62     | 373     |

**Acurácia:** 0.74  
**F1 ponderado:** 0.76

📌 O modelo teve melhor desempenho com as 5 melhores features, garantindo alta sensibilidade na classe de interesse.

---

## 7. Teste com Dados de Validação

### 🎯 Resultados:

| Classe | Precision | Recall | F1-score | Suporte |
|--------|-----------|--------|----------|---------|
| 0      | 0.93      | 0.81   | 0.87     | 16      |
| 1      | 0.57      | 0.80   | 0.67     | 5       |

- 📦 Acurácia: 81%  
- 📦 F1 ponderado: 0.82  
- 🔍 🔗 [Link da matriz de confusão]  
- 📄 🔗 [CSV com resultados da validação]

⚠️ Amostra pequena (n=21), ainda assim mostra robustez com recall alto na classe 1.

---

## 8. Conclusão Estratégica

O modelo final Random Forest com SMOTE atingiu 78% de recall na classe de churn, com bom equilíbrio geral.

Variáveis como contrato mensal, permanência, gastos diários e tipo de internet (Fibra Óptica) são os principais influenciadores do cancelamento.

O modelo pode ser integrado em pipelines de retenção e campanhas direcionadas para evitar evasões futuras.

Apesar dos bons resultados, ações de retenção baseadas nas previsões devem considerar o trade-off entre precisão e custo operacional.

---

## 📂 Recursos Visuais

- 🔗 [Matriz de Confusão]  
- 🔗 [Curva ROC AUC]  
- 🔗 [Curva Precisão x Recall]  
- 🔗 [Features Importances (Gráfico)]  
- 🔗 [Curva de Aprendizado (Overfitting/Underfitting)]


---

Arquitetura_Projeto.md
# 🏗️ Arquitetura do Projeto – Previsão de Evasão de Clientes (Churn)

---

## 📁 Estrutura de Pastas e Arquivos

# 📁 Estrutura de Diretórios do Projeto `churn_prediction_project`

```bash
churn_prediction_project/
│
├── data/                             # Dados utilizados no projeto
│   ├── raw/                          # 📦 Dados brutos originais
│   ├── processed/                    # 🧹 Dados tratados e codificados
│   ├── balanced/                     # ⚖️ Dados balanceados (oversampling e undersampling)
│   └── validation/                   # ✅ Dados de validação final
│
├── notebooks/
│   └── churn_modeling.ipynb          # 📒 Notebook principal com todo o pipeline de modelagem
│
├── reports/
│   ├── graphics/                     # 📊 Imagens: matriz, curvas ROC, precisão x recall, features
│   ├── metrics/                      # 📑 Relatórios em CSV (classification report, cross-validation, etc.)
│   └── model_test_results/           # 🧪 Resultados do teste com novos dados
│
├── models/
│   └── final_model.pkl               # 🧠 Modelo treinado salvo com pipeline
│
├── src/                              # 📦 Código-fonte modular
│   ├── preprocessing.py              # 🔧 Funções de tratamento e codificação
│   ├── modeling.py                   # 🤖 Treinamento, avaliação e tuning de modelos
│   └── utils.py                      # 🧰 Funções auxiliares (plotagem, métricas, etc.)
│
├── requirements.txt                  # 📦 Lista de dependências e bibliotecas utilizadas
├── README.md                         # 🗂️ Documentação geral e apresentação do projeto
├── Relatorio_Tecnico.md              # 📘 Relatório técnico completo
└── Arquitetura_Projeto.md            # 🧱 Estrutura e organização do projeto

---

## 🧱 Componentes do Projeto

### 1. **Coleta e Leitura dos Dados**
- Fonte: Arquivo CSV local (ou importado via URL)
- Biblioteca: `pandas`

### 2. **Tratamento e Engenharia de Dados**
- Remoção de colunas redundantes (ex: `gastos_mensais`)
- Codificação: `OneHotEncoder` (via `pandas.get_dummies`)
- Armazenamento do dataset processado em `/data/processed/`

### 3. **Exploração e Visualização**
- Bibliotecas: `seaborn`, `matplotlib`
- Gráficos:
  - Correlação
  - Boxplots
  - Tempo de contrato × Cancelamento
  - Total gasto × Cancelamento

### 4. **Balanceamento dos Dados**
- Oversampling: `SMOTE`
- Undersampling: `NearMiss`
- Aplicado dentro de `Pipeline` com `imbpipeline`

### 5. **Modelagem**
- Algoritmos:
  - `DecisionTreeClassifier`
  - `RandomForestClassifier`
- Biblioteca: `sklearn.ensemble`, `sklearn.tree`
- Ajustes: `class_weight=balanced`, `max_depth=10`

### 6. **Validação e Avaliação**
- Separação: `train_test_split` (75/25)
- Métricas:
  - `confusion_matrix`
  - `classification_report`
  - `roc_auc_score`
  - `precision_recall_curve`
- Visualizações:
  - Matriz de confusão
  - Curva ROC
  - Curva Precisão x Recall
  - Learning Curve (`sklearn.model_selection.learning_curve`)

### 7. **Tuning de Hiperparâmetros**
- RandomizedSearchCV ✅
- GridSearchCV (não adotado)
- Estratégia de cross-validation com `StratifiedKFold`
- Armazenamento dos resultados em `/reports/metrics/`

### 8. **Exportação e Produção**
- Modelo salvo via `pickle` (`final_model.pkl`)
- Pipeline salva inclui:
  - Pré-processamento
  - Codificação
  - Modelo
- Variáveis salvas:
  - `features_names_in_`
  - `pipeline` final

---

## ⚙️ Tecnologias Utilizadas

| Categoria                 | Tecnologia / Biblioteca        |
|---------------------------|-------------------------------|
| Manipulação de Dados      | `pandas`, `numpy`             |
| Visualização              | `matplotlib`, `seaborn`       |
| Modelagem Preditiva       | `sklearn`, `imblearn`         |
| Balanceamento de Dados    | `SMOTE`, `NearMiss`           |
| Otimização de Modelos     | `GridSearchCV`, `RandomizedSearchCV` |
| Métricas e Avaliação      | `confusion_matrix`, `roc_auc_score`, `precision_recall_curve` |
| Exportação de Modelo      | `pickle`                      |
| Ambiente de Desenvolvimento | Google Colab / Jupyter Notebook |

---

## 🔐 Persistência de Artefatos

- `final_model.pkl` → Contém modelo, pipeline e colunas
- Dados balanceados e tratados armazenados em CSVs
- Métricas salvas como CSVs
- Gráficos salvos em `/reports/graphics/` com nomes padronizados

---

## 🧪 Teste do Modelo com Dados Finais

- Utilizado conjunto `/data/validation/`
- Resultados armazenados em:
  - Matriz: `/reports/graphics/matriz_validacao.png`
  - CSV: `/reports/model_test_results/teste_validacao.csv`

---

## 📌 Observações Finais

- O pipeline é **100% reprodutível** e pode ser reaplicado com novos dados.
- A estrutura de projeto foi pensada para **facilitar reuso, manutenção e portabilidade.**
- Modelo encapsula todas as etapas, eliminando necessidade de retratamento fora do pipeline.

---


REQUERIMENTOS

pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.4
seaborn==0.13.2
scikit-learn==1.4.2
imbalanced-learn==0.12.0
pickle-mixin==1.0.2



