<div align="center">
  <img src="https://raw.githubusercontent.com/Siiqueira/TelecomX_BR_II/refs/heads/main/data/results/img/Banner.jpg" width="1200" height="370" />
</div>  

# 📊 **Relatório Técnico – Análise do Modelo de Previsão de Evasão de Clientes (Churn)**  

---

<div align="center"> 
    <p><b>Paleta usada em todo o projeto</b></p>
  <img src="https://raw.githubusercontent.com/Siiqueira/TelecomX_BR_II/refs/heads/main/data/results/img/paleta.png" width="300" height="200" />
</div>  

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

⚠️ *Os dados estão desbalanceados. Isso pode causar viés nos modelos preditivos. Técnicas de oversampling e undersampling foram testadas para mitigar esse problema.*

---

## 2. Análise Exploratória e Correlações

### Principais descobertas:

- **Gastos Mensais** e **Gastos Diários** têm **correlação perfeita**. Decidimos remover `gastos_mensais`.
- Contrato Mensal tem forte correlação com cancelamento (**+40%**).
- Clientes com maior permanência têm menor chance de cancelar (**correlação negativa de -35%**).
- Fibra Óptica está correlacionada com gastos elevados e maior taxa de churn.
- Clientes com contratos mais curtos tendem a acumular menos gastos e cancelar mais cedo.

### 📈 **Gráficos:**
#### **Tempo de Contrato x Cancelamento**  
<div align="center">
  <img src="https://raw.githubusercontent.com/Siiqueira/TelecomX_BR_II/refs/heads/main/data/results/img/cancelamento_tempo_contrato.png" width="500" height="400" />
</div>
  

#### **Total de Gasto x Cancelamento**  
<div align="center">
  <img src="https://raw.githubusercontent.com/Siiqueira/TelecomX_BR_II/refs/heads/main/data/results/img/cancelamento_total_gastos.png" width="500" height="400" />
</div>

#### **Gráfico de Correlação**
<div align="center">
  <img src="https://raw.githubusercontent.com/Siiqueira/TelecomX_BR_II/refs/heads/main/data/results/img/matriz_correlacao.png" width="1000" height="600" />
</div>

---

## 3. Modelagem Inicial

### Separação dos dados:

- **Dados de treino:** (5608, 26)
- **Dados de teste:** (1403, 26)

### Modelos testados:

**Decision Tree Classifier**
```python
DecisionTreeClassifier(
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
```
**Random Forest Classifier**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
```

## 4. Avaliação dos Modelos

### 🔢 Métricas Comparativas

| Modelo         |Precision | Recall (Classe 1) | F1 (Classe 1) | Accuracy |
|:--------------:|:--------:|:-----------------:|:-------------:|:--------:|
| Decision Tree  |   0.48   | 0.69              | 0.57          | 0.72     |
| Random Forest  |   0.54   | 0.71              | 0.61          | 0.76     |

### 📉 Matriz de Confusão

**Decision Tree x Random Forest**  
<div align="left">
  <img src="https://raw.githubusercontent.com/Siiqueira/TelecomX_BR_II/refs/heads/main/data/results/img/matriz_confusao_comparacao.png" width="700" height="500" />
</div>

 > Random Forest foi superior em quase todas as métricas, principalmente na identificação correta de clientes que não cancelam (classe 0).
 > A classe 1 (cancelamento) ainda apresenta desafios.

### 📈 Curva ROC AUC  

**Decision Tree x Random Forest** 
<div align="left">
  <img src="https://raw.githubusercontent.com/Siiqueira/TelecomX_BR_II/refs/heads/main/data/results/img/curva_roc_comparacao.png" width="700" height="500" />
</div>

- Decision Tree: AUC = 0.73  
- Random Forest: AUC = 0.82

### 📈 Curva Precisão x Recall

**Decision Tree x Random Forest**  
<div align="left">
  <img src="https://raw.githubusercontent.com/Siiqueira/TelecomX_BR_II/refs/heads/main/data/results/img/curva_pr_comparacao.png" width="700" height="500" />
</div>
- Decision Tree: Avg. Precision = 0.44  
- Random Forest: Avg. Precision = 0.59

### 📄 Classification Report

[**Comparação: Decision Tree x Random Forest (CSV)**](https://github.com/Siiqueira/TelecomX_BR_II/blob/main/data/results/csv/classification_report_comparacao.csv)  

> Random Forest teve melhor desempenho para a classe de interesse (churn).  
> F1-Score e recall da classe 1 foram superiores ao Decision Tree.

### 🔁 Validação Cruzada

[**Comparação: Decision Tree x Random Forest** (CSV)](https://github.com/Siiqueira/TelecomX_BR_II/blob/main/data/results/csv/validacao_cruzada_comparacao.csv)  

> Resultados mais estáveis e consistentes com Random Forest.  
> Menor variância e melhor generalização.

---

## 5. Balanceamento dos Dados

**Técnicas utilizadas:**

- Oversampling (SMOTE) ✅ Escolhido  
- Undersampling (NearMiss (Version = 3)
  

📄 [**Oversampling (CSV)**](https://github.com/Siiqueira/TelecomX_BR_II/blob/main/data/results/csv/validacao_cruzada_oversampling.csv)  
📄 [**Undersampling (CSV)**](https://github.com/Siiqueira/TelecomX_BR_II/blob/main/data/results/csv/validacao_cruzada_undersampling.csv)  

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

- 📦 **Acurácia: 81%**    
- 📦 **F1 ponderado: 0.82**    
- 📄  [**Tabela com Teste (CSV)**](https://github.com/Siiqueira/TelecomX_BR_II/blob/main/data/processed/df_validacao.csv)
- 📈 **Matriz de confusão Validação**
<div align="left">
  <img src="https://raw.githubusercontent.com/Siiqueira/TelecomX_BR_II/refs/heads/main/data/results/img/matriz_confusao_testando_modelo.png" width="500" height="500" />
</div>


⚠️ Amostra pequena (n=21), ainda assim mostra robustez com recall alto na classe 1.

---

## 8. Conclusão Estratégica

O modelo final Random Forest com SMOTE atingiu 78% de recall na classe de churn, com bom equilíbrio geral.

Variáveis como contrato mensal, permanência, gastos diários e tipo de internet (Fibra Óptica) são os principais influenciadores do cancelamento.

O modelo pode ser integrado em pipelines de retenção e campanhas direcionadas para evitar evasões futuras.

Apesar dos bons resultados, ações de retenção baseadas nas previsões devem considerar o trade-off entre precisão e custo operacional.

---

## 📂 **Gráficos do modelo final**


-  [**Gráfico de Matriz de Confusão**](https://github.com/Siiqueira/TelecomX_BR_II/blob/main/data/results/img/matriz_confusao_modelo_final.png)    
-  [**Gráfico de Curva ROC AUC**](https://github.com/Siiqueira/TelecomX_BR_II/blob/main/data/results/img/curva_roc_modelo_final.png)    
-  [**Gráfico deCurva Precisão x Recall (AP)**](https://github.com/Siiqueira/TelecomX_BR_II/blob/main/data/results/img/curva_precision_recall_modelo_final.png)    
-  [**Gráfico com as Features Importances**](https://github.com/Siiqueira/TelecomX_BR_II/blob/main/data/results/img/feature_importances_modelo_final.png)    
-  [**Gráfico da Curva de Aprendizado (Overfitting/Underfitting)**](https://github.com/Siiqueira/TelecomX_BR_II/blob/main/data/results/img/learning_curve.png)  


---




