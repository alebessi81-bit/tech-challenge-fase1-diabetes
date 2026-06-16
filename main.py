"""
Tech Challenge Fase 1 — Diagnóstico de Diabetes
================================================
Pipeline completo: pré-processamento → treino → avaliação

Uso:
    python main.py                         # roda pipeline completo
    python main.py --etapa preprocessing   # só pré-processamento
    python main.py --etapa treino          # só treino
    python main.py --etapa avaliacao       # só avaliação
    python main.py --dados caminho.csv     # dataset customizado
"""

import argparse
import sys
import os

# Adiciona src/ ao path para importar os módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import pipeline_completo
from train import pipeline_treino, carregar_modelos
from evaluate import pipeline_avaliacao

import pandas as pd


# ── Configurações padrão ─────────────────────────────────────────────────────
CAMINHO_DADOS    = 'data/diabetes.csv'
PASTA_PROCESSADO = 'data/processed'
PASTA_MODELOS    = 'data/models'
PASTA_GRAFICOS   = 'notebooks'


def etapa_preprocessing(caminho_csv):
    print("\n" + "="*55)
    print("  ETAPA 1 — PRÉ-PROCESSAMENTO")
    print("="*55)
    pipeline_completo(caminho_csv, PASTA_PROCESSADO)
    print("\n✅ Pré-processamento concluído!\n")


def etapa_treino():
    print("\n" + "="*55)
    print("  ETAPA 2 — TREINAMENTO DOS MODELOS")
    print("="*55)
    lr, rf, X_train, X_val, X_test, y_train, y_val, y_test = pipeline_treino(
        PASTA_PROCESSADO, PASTA_MODELOS
    )
    print("\n✅ Treinamento concluído!\n")
    return lr, rf, X_test, y_test


def etapa_avaliacao(lr=None, rf=None, X_test=None, y_test=None):
    print("\n" + "="*55)
    print("  ETAPA 3 — AVALIAÇÃO DOS MODELOS")
    print("="*55)

    # Se não recebeu modelos e dados, carrega do disco
    if lr is None or rf is None:
        lr, rf = carregar_modelos(PASTA_MODELOS)

    if X_test is None or y_test is None:
        X_test = pd.read_csv(f'{PASTA_PROCESSADO}/X_test.csv')
        y_test = pd.read_csv(f'{PASTA_PROCESSADO}/y_test.csv').values.ravel()

        # Corrigir NaN remanescentes
        for col in X_test.columns:
            X_test[col] = X_test[col].fillna(X_test[col].median())

    modelos = {
        'Regressão Logística': lr,
        'Random Forest': rf
    }

    resultados = pipeline_avaliacao(modelos, X_test, y_test, PASTA_GRAFICOS)

    print("\n" + "="*55)
    print("  RESUMO FINAL")
    print("="*55)
    for r in resultados:
        print(f"  {r['nome']:<25} Recall={r['recall']:.4f} | F1={r['f1']:.4f} | Acc={r['accuracy']:.4f}")

    melhor = max(resultados, key=lambda r: r['recall'])
    print(f"\n  🏆 Melhor modelo: {melhor['nome']} (Recall={melhor['recall']:.4f})")
    print("\n✅ Avaliação concluída!\n")


def pipeline_completo_cli(caminho_csv):
    print("\n" + "="*55)
    print("  TECH CHALLENGE FASE 1 — DIAGNÓSTICO DE DIABETES")
    print("="*55)
    print(f"  Dataset: {caminho_csv}")
    print("="*55)

    etapa_preprocessing(caminho_csv)
    lr, rf, X_test, y_test = etapa_treino()
    etapa_avaliacao(lr, rf, X_test, y_test)

    print("="*55)
    print("  PIPELINE COMPLETO FINALIZADO COM SUCESSO!")
    print("="*55 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Tech Challenge Fase 1 — Diagnóstico de Diabetes'
    )
    parser.add_argument(
        '--etapa',
        choices=['preprocessing', 'treino', 'avaliacao', 'completo'],
        default='completo',
        help='Etapa a executar (padrão: completo)'
    )
    parser.add_argument(
        '--dados',
        default=CAMINHO_DADOS,
        help=f'Caminho para o CSV do dataset (padrão: {CAMINHO_DADOS})'
    )

    args = parser.parse_args()

    if args.etapa == 'preprocessing':
        etapa_preprocessing(args.dados)
    elif args.etapa == 'treino':
        etapa_treino()
    elif args.etapa == 'avaliacao':
        etapa_avaliacao()
    elif args.etapa == 'completo':
        pipeline_completo_cli(args.dados)


if __name__ == '__main__':
    main()
