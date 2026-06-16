# 🩺 Tech Challenge Fase 1 — Diagnóstico de Diabetes

> Solução de Machine Learning para diagnóstico de diabetes desenvolvida como entrega do Tech Challenge Fase 1 do programa PosTech FIAP — IA para Devs.

---

## 📋 Descrição do Projeto

Este projeto implementa um sistema inteligente de suporte ao diagnóstico médico, focado na **classificação de diabetes** a partir de dados clínicos estruturados.

O pipeline completo inclui:
- Análise Exploratória de Dados (EDA)
- Pré-processamento e limpeza dos dados
- Treinamento e comparação de modelos de Machine Learning
- Avaliação com métricas clínicas (foco em Recall)
- Interpretabilidade com Feature Importance e SHAP

---

## 📂 Estrutura do Projeto

```
tech-challenge-fase1-diabetes/
├── data/
│   ├── diabetes.csv              # Dataset original
│   ├── processed/                # Dados pré-processados (gerado automaticamente)
│   └── models/                   # Modelos treinados (gerado automaticamente)
├── notebooks/
│   ├── 01_eda.ipynb              # Análise Exploratória de Dados
│   ├── 02_preprocessing.ipynb    # Pré-processamento
│   ├── 03_modelagem.ipynb        # Treinamento e avaliação
│   └── 04_interpretabilidade.ipynb # Feature Importance e SHAP
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Módulo de pré-processamento
│   ├── train.py                  # Módulo de treinamento
│   └── evaluate.py               # Módulo de avaliação
├── main.py                       # Ponto de entrada CLI
├── Dockerfile                    # Containerização
├── requirements.txt              # Dependências Python
└── README.md
```

---

## 🗃️ Dataset

**Pima Indians Diabetes Dataset**
- Fonte: [Kaggle — mathchi/diabetes-data-set](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data)
- 768 pacientes, 8 features clínicas + 1 variável alvo (`Outcome`)
- Features: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`
- Alvo: `Outcome` (0 = sem diabetes, 1 = com diabetes)

---

## ⚙️ Pré-requisitos

- Python 3.11+
- pip

---

## 🚀 Instalação e Execução

### 1. Clonar o repositório

```bash
git clone https://github.com/alebessi81-bit/desafio-tecnologico-fase1-diabetes.git
cd desafio-tecnologico-fase1-diabetes
```

### 2. Criar ambiente virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Baixar o dataset

Baixe o arquivo `diabetes.csv` em:
https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data

Coloque o arquivo em `data/diabetes.csv`.

### 5. Executar o pipeline completo

```bash
python main.py
```

### Opções da CLI

```bash
# Pipeline completo (padrão)
python main.py

# Apenas pré-processamento
python main.py --etapa preprocessing

# Apenas treinamento
python main.py --etapa treino

# Apenas avaliação
python main.py --etapa avaliacao

# Dataset em caminho customizado
python main.py --dados caminho/para/dataset.csv
```

---

## 🐳 Execução com Docker

### Build da imagem

```bash
docker build -t diabetes-ml .
```

### Rodar o pipeline

```bash
docker run diabetes-ml
```

### Rodar etapa específica

```bash
docker run diabetes-ml python main.py --etapa avaliacao
```

---

## 📊 Resultados

| Modelo               | Accuracy | Recall ★ | F1-score | Precision |
|----------------------|----------|-----------|----------|-----------|
| Regressão Logística  | 0.7759   | 0.5122    | 0.6176   | 0.7778    |
| Random Forest        | 0.7414   | 0.4878    | 0.5714   | 0.6897    |

> ★ Recall é a métrica principal. Em diagnóstico médico, minimizar falsos negativos (diabéticos não detectados) é prioritário.

**Modelo escolhido: Regressão Logística** — melhor Recall e F1-score, além de maior interpretabilidade clínica.

---

## 🔍 Interpretabilidade

As features mais relevantes para o diagnóstico, segundo SHAP e Feature Importance:

1. **Glucose** — maior preditor de diabetes (SHAP médio: 0.9498)
2. **BMI** — segundo fator mais relevante (SHAP médio: 0.6511)
3. **Pregnancies** — terceiro fator (SHAP médio: 0.4182)

---

## ⚠️ Aviso Clínico

Este modelo é uma **ferramenta de triagem e apoio**, não um sistema de diagnóstico definitivo. O(a) médico(a) deve sempre ter a palavra final no diagnóstico clínico.

---

## 🛠️ Tecnologias Utilizadas

- Python 3.11
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- shap
- jupyter

---

## 👤 Autor

**Alessandra Bessi**
PosTech FIAP — IA para Devs
Tech Challenge Fase 1
