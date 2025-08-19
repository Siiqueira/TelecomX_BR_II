\# 🏗️ Arquitetura do Projeto – Previsão de Evasão de Clientes (Churn)



---


\## 📁 Estrutura de Pastas e Arquivos



\# 📁 Estrutura de Diretórios do Projeto `churn\_prediction\_project`



```bash

churn\_prediction\_project/

│

├── data/                             # Dados utilizados no projeto

│   ├── raw/                          # 📦 Dados brutos originais

│   ├── processed/                    # 🧹 Dados tratados e codificados

│   ├── balanced/                     # ⚖️ Dados balanceados (oversampling e undersampling)

│   └── validation/                   # ✅ Dados de validação final

│

├── notebooks/

│   └── churn\_modeling.ipynb          # 📒 Notebook principal com todo o pipeline de modelagem

│

├── reports/

│   ├── graphics/                     # 📊 Imagens: matriz, curvas ROC, precisão x recall, features

│   ├── metrics/                      # 📑 Relatórios em CSV (classification report, cross-validation, etc.)

│   └── model\_test\_results/           # 🧪 Resultados do teste com novos dados

│

├── models/

│   └── final\_model.pkl               # 🧠 Modelo treinado salvo com pipeline

│

├── src/                              # 📦 Código-fonte modular

│   ├── preprocessing.py              # 🔧 Funções de tratamento e codificação

│   ├── modeling.py                   # 🤖 Treinamento, avaliação e tuning de modelos

│   └── utils.py                      # 🧰 Funções auxiliares (plotagem, métricas, etc.)

│

├── requirements.txt                  # 📦 Lista de dependências e bibliotecas utilizadas

├── README.md                         # 🗂️ Documentação geral e apresentação do projeto

├── Relatorio\_Tecnico.md              # 📘 Relatório técnico completo

└── Arquitetura\_Projeto.md            # 🧱 Estrutura e organização do projeto

```

---



\## 🧱 Componentes do Projeto



\### 1. \*\*Coleta e Leitura dos Dados\*\*

\- Fonte: Arquivo CSV local (ou importado via URL)

\- Biblioteca: `pandas`



\### 2. \*\*Tratamento e Engenharia de Dados\*\*

\- Remoção de colunas redundantes (ex: `gastos\_mensais`)

\- Codificação: `OneHotEncoder` (via `pandas.get\_dummies`)

\- Armazenamento do dataset processado em `/data/processed/`



\### 3. \*\*Exploração e Visualização\*\*

\- Bibliotecas: `seaborn`, `matplotlib`

\- Gráficos:

&nbsp; - Correlação

&nbsp; - Boxplots

&nbsp; - Tempo de contrato × Cancelamento

&nbsp; - Total gasto × Cancelamento



\### 4. \*\*Balanceamento dos Dados\*\*

\- Oversampling: `SMOTE`

\- Undersampling: `NearMiss`

\- Aplicado dentro de `Pipeline` com `imbpipeline`



\### 5. \*\*Modelagem\*\*

\- Algoritmos:

&nbsp; - `DecisionTreeClassifier`

&nbsp; - `RandomForestClassifier`

\- Biblioteca: `sklearn.ensemble`, `sklearn.tree`

\- Ajustes: `class\_weight=balanced`, `max\_depth=10`



\### 6. \*\*Validação e Avaliação\*\*

\- Separação: `train\_test\_split` (75/25)

\- Métricas:

&nbsp; - `confusion\_matrix`

&nbsp; - `classification\_report`

&nbsp; - `roc\_auc\_score`

&nbsp; - `precision\_recall\_curve`

\- Visualizações:

&nbsp; - Matriz de confusão

&nbsp; - Curva ROC

&nbsp; - Curva Precisão x Recall

&nbsp; - Learning Curve (`sklearn.model\_selection.learning\_curve`)



\### 7. \*\*Tuning de Hiperparâmetros\*\*

\- RandomizedSearchCV ✅

\- GridSearchCV (não adotado)

\- Estratégia de cross-validation com `StratifiedKFold`

\- Armazenamento dos resultados em `/reports/metrics/`



\### 8. \*\*Exportação e Produção\*\*

\- Modelo salvo via `pickle` (`final\_model.pkl`)

\- Pipeline salva inclui:

&nbsp; - Pré-processamento

&nbsp; - Codificação

&nbsp; - Modelo

\- Variáveis salvas:

&nbsp; - `features\_names\_in\_`

&nbsp; - `pipeline` final



---



\## ⚙️ Tecnologias Utilizadas



| Categoria                 | Tecnologia / Biblioteca        |

|---------------------------|-------------------------------|

| Manipulação de Dados      | `pandas`, `numpy`             |

| Visualização              | `matplotlib`, `seaborn`       |

| Modelagem Preditiva       | `sklearn`, `imblearn`         |

| Balanceamento de Dados    | `SMOTE`, `NearMiss`           |

| Otimização de Modelos     | `GridSearchCV`, `RandomizedSearchCV` |

| Métricas e Avaliação      | `confusion\_matrix`, `roc\_auc\_score`, `precision\_recall\_curve` |

| Exportação de Modelo      | `pickle`                      |

| Ambiente de Desenvolvimento | Google Colab / Jupyter Notebook |



---



\## 🔐 Persistência de Artefatos



\- `final\_model.pkl` → Contém modelo, pipeline e colunas

\- Dados balanceados e tratados armazenados em CSVs

\- Métricas salvas como CSVs

\- Gráficos salvos em `/reports/graphics/` com nomes padronizados



---



\## 🧪 Teste do Modelo com Dados Finais



\- Utilizado conjunto `/data/validation/`

\- Resultados armazenados em:

&nbsp; - Matriz: `/reports/graphics/matriz\_validacao.png`

&nbsp; - CSV: `/reports/model\_test\_results/teste\_validacao.csv`



---



\## 📌 Observações Finais



\- O pipeline é \*\*100% reprodutível\*\* e pode ser reaplicado com novos dados.

\- A estrutura de projeto foi pensada para \*\*facilitar reuso, manutenção e portabilidade.\*\*

\- Modelo encapsula todas as etapas, eliminando necessidade de retratamento fora do pipeline.



---









