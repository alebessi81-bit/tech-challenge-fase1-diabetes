import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os


COLUNAS_SUSPEITAS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


def carregar_dados(caminho: str) -> pd.DataFrame:
    """Carrega o dataset CSV."""
    df = pd.read_csv(caminho)
    print(f"[preprocessing] Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


def tratar_zeros_suspeitos(df: pd.DataFrame) -> pd.DataFrame:
    """Substitui zeros fisiologicamente impossíveis por NaN e imputa pela mediana."""
    df = df.copy()
    df[COLUNAS_SUSPEITAS] = df[COLUNAS_SUSPEITAS].replace(0, np.nan)
    for col in COLUNAS_SUSPEITAS:
        mediana = df[col].median()
        df[col] = df[col].fillna(mediana)
        print(f"[preprocessing] {col}: zeros imputados pela mediana ({mediana:.2f})")
    return df


def separar_features_alvo(df: pd.DataFrame):
    """Separa features (X) e variável alvo (y)."""
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y


def dividir_dados(X, y, test_size=0.30, val_size=0.50, random_state=42):
    """Divide em treino (70%), validação (15%) e teste (15%)."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    print(f"[preprocessing] Treino: {X_train.shape[0]} | Validação: {X_val.shape[0]} | Teste: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def escalonar(X_train, X_val, X_test):
    """Aplica StandardScaler — fit apenas no treino."""
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_s   = pd.DataFrame(scaler.transform(X_val),       columns=X_val.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
    print("[preprocessing] Escalonamento aplicado com StandardScaler")
    return X_train_s, X_val_s, X_test_s, scaler


def salvar_dados_processados(X_train, X_val, X_test, y_train, y_val, y_test, scaler, pasta='data/processed'):
    """Salva os dados processados e o scaler em disco."""
    os.makedirs(pasta, exist_ok=True)
    X_train.to_csv(f'{pasta}/X_train.csv', index=False)
    X_val.to_csv(f'{pasta}/X_val.csv',     index=False)
    X_test.to_csv(f'{pasta}/X_test.csv',   index=False)
    y_train.to_csv(f'{pasta}/y_train.csv', index=False)
    y_val.to_csv(f'{pasta}/y_val.csv',     index=False)
    y_test.to_csv(f'{pasta}/y_test.csv',   index=False)
    with open(f'{pasta}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"[preprocessing] Dados salvos em '{pasta}/'")


def pipeline_completo(caminho_csv: str, pasta_saida='data/processed'):
    """Executa o pipeline completo de pré-processamento."""
    df = carregar_dados(caminho_csv)
    df = tratar_zeros_suspeitos(df)
    X, y = separar_features_alvo(df)
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_dados(X, y)
    X_train_s, X_val_s, X_test_s, scaler = escalonar(X_train, X_val, X_test)
    salvar_dados_processados(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, scaler, pasta_saida)
    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, scaler
