import pickle
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def carregar_dados_processados(pasta='data/processed'):
    """Carrega os dados processados do disco."""
    X_train = pd.read_csv(f'{pasta}/X_train.csv')
    X_val   = pd.read_csv(f'{pasta}/X_val.csv')
    X_test  = pd.read_csv(f'{pasta}/X_test.csv')
    y_train = pd.read_csv(f'{pasta}/y_train.csv').values.ravel()
    y_val   = pd.read_csv(f'{pasta}/y_val.csv').values.ravel()
    y_test  = pd.read_csv(f'{pasta}/y_test.csv').values.ravel()

    # Corrigir NaN remanescentes
    for col in X_train.columns:
        mediana = X_train[col].median()
        X_train[col] = X_train[col].fillna(mediana)
        X_val[col]   = X_val[col].fillna(mediana)
        X_test[col]  = X_test[col].fillna(mediana)

    print(f"[train] Dados carregados — Treino: {X_train.shape} | Val: {X_val.shape} | Teste: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def treinar_regressao_logistica(X_train, y_train):
    """Treina e retorna a Regressão Logística."""
    print("[train] Treinando Regressão Logística...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    print("[train] Regressão Logística treinada!")
    return lr


def treinar_random_forest(X_train, y_train):
    """Treina e retorna o Random Forest."""
    print("[train] Treinando Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    print("[train] Random Forest treinado!")
    return rf


def salvar_modelos(lr, rf, pasta='data/models'):
    """Salva os modelos treinados em disco."""
    os.makedirs(pasta, exist_ok=True)
    with open(f'{pasta}/logistic_regression.pkl', 'wb') as f:
        pickle.dump(lr, f)
    with open(f'{pasta}/random_forest.pkl', 'wb') as f:
        pickle.dump(rf, f)
    print(f"[train] Modelos salvos em '{pasta}/'")


def carregar_modelos(pasta='data/models'):
    """Carrega os modelos salvos do disco."""
    with open(f'{pasta}/logistic_regression.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open(f'{pasta}/random_forest.pkl', 'rb') as f:
        rf = pickle.load(f)
    print(f"[train] Modelos carregados de '{pasta}/'")
    return lr, rf


def pipeline_treino(pasta_dados='data/processed', pasta_modelos='data/models'):
    """Executa o pipeline completo de treinamento."""
    X_train, X_val, X_test, y_train, y_val, y_test = carregar_dados_processados(pasta_dados)
    lr = treinar_regressao_logistica(X_train, y_train)
    rf = treinar_random_forest(X_train, y_train)
    salvar_modelos(lr, rf, pasta_modelos)
    return lr, rf, X_train, X_val, X_test, y_train, y_val, y_test
