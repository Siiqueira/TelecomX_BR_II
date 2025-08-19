
<p align="center">
  <img src="/banner.jpg" alt="Banner" tyle="width:100%;">
</p>


# Projeto: **Previsão de Cancelamento de Clientes (Churn) - TelecomX BR**  

[TelecomX BR - Parte I](https://github.com/Siiqueira/TelecomX_BR)

## 🎯 Missão do Cientista de Dados
Desenvolver um pipeline de machine learning completo para prever a evasão (churn) de clientes da TelecomX_BR, permitindo ações proativas de retenção por parte da empresa.  

**Respostas que vamos responder:**

> - Quem são os clientes com maior risco de evasão?
> - Quais variáveis mais Influenciam esse comportamento?
> - Que tipo de perfil a empresa precisa manter mais próximo?

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
📦 TelecomX_BR_II/
├── data/
│   ├── raw/                         # Dados brutos
│   ├── processed/                   # Dados tratados
│   └── results/                       
|       └── csv/                     # CSVs
|       └── img/                     # Gráficos gerados  
│
├── models/
│   └── model_telecomx_BR.pkl        # Modelo final salvo
│
├── notebooks/
│   └──TelecomX_II.ipynb             # Notebook principal do projeto
│
├── src/
│   └── pipeline/                    # Lógica do pipeline (pré-processamento, modelagem, etc.)
│
├── README.md                        # Documentação principal do projeto
├── Relatorio_Tecnico.md             # Relatório técnico completo
├── Arquitetura_Projeto.md           # Estrutura e decisões arquiteturais
├── LICENSE                          # Licença de uso (MIT)
├── .gitignore                       # Arquivos e pastas ignorados pelo Git   
└── requirements.txt                 # Bibliotecas do modelo
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
```
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
- 

### 🗂️ Links Importantes

- 📈 Gráficos de correlação e distribuição  
- 📊 Matriz de confusão  
- 📊 Curva ROC AUC  
- 📊 Curva Precisão x Recall  
- 📊 Features Importances (CSV + gráfico)  
- 📊 Resultados de validação cruzada  
- 📊 Comparações entre modelos    


## 📦 Como Usar

Clone o repositório:  
```
git clone https://github.com/Siiqueira/TelecomX_BR_II.git
``` 

Instale as dependências:  
```
pip install -r requirements.txt
```

Execute os notebooks disponíveis na pasta notebooks/.

Avalie o modelo final ou carregue o modelo .pkl com o seguinte código:

Import pickle  
```
with open('models/modelo_final.pkl', 'rb') as f:
    model = pickle.load(f)
```
---

## 📌 Observações

Este projeto é voltado para análise de churn com dados sintéticos e objetivos didáticos.

Para implementação em produção, recomenda-se:

- Monitoramento contínuo de métricas;
- Coleta de novas variáveis comportamentais;
- Atualização periódica do modelo.


## 🧠 Conclusão

Este projeto entregou uma solução preditiva eficiente para o problema de churn, com foco em clientes com maior risco de cancelamento, variáveis mais influentes e estratégias práticas de retenção. O modelo Random Forest com SMOTE demonstrou robustez, generalização e capacidade de auxiliar a empresa na tomada de decisões estratégicas.

### Conclusão Estratégica

Clientes com contratos mensais, baixa permanência, e que utilizam fibra óptica representam o maior risco de churn.

Estratégias de retenção ativa devem focar nesses perfis.

O modelo, mesmo diante de dados desbalanceados, apresenta capacidade robusta de generalização e é adequado para uso em produção.

---

#### 📮 **Contato**

> 👨‍💻 Desenvolvido por: Ellan Alves  
> 📧 Email: ynvestellan@gmail.com  
> 🔗 LinkedIn: https//www.linkedin.com/in/ellan-alves-dados
---
