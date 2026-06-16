import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import os


def avaliar_modelo(nome, modelo, X, y, conjunto='Teste'):
    """Avalia um modelo e retorna as métricas principais."""
    y_pred = modelo.predict(X)
    y_prob = modelo.predict_proba(X)[:, 1] if hasattr(modelo, 'predict_proba') else None

    acc  = accuracy_score(y, y_pred)
    rec  = recall_score(y, y_pred)
    f1   = f1_score(y, y_pred)
    prec = precision_score(y, y_pred)
    auc  = roc_auc_score(y, y_prob) if y_prob is not None else None

    print(f"\n=== {nome} | {conjunto} ===")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Recall:    {rec:.4f}  <- métrica principal")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    if auc:
        print(f"  ROC-AUC:   {auc:.4f}")

    return {'nome': nome, 'accuracy': acc, 'recall': rec, 'f1': f1, 'precision': prec, 'auc': auc}


def relatorio_completo(nome, modelo, X, y):
    """Imprime o classification report completo."""
    y_pred = modelo.predict(X)
    print(f"\n=== RELATÓRIO COMPLETO — {nome} ===")
    print(classification_report(y, y_pred, target_names=['Sem Diabetes', 'Com Diabetes']))


def plotar_matriz_confusao(modelos_dict, X, y, pasta_saida='notebooks'):
    """Plota matrizes de confusão lado a lado."""
    n = len(modelos_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (nome, modelo) in zip(axes, modelos_dict.items()):
        cm = confusion_matrix(y, modelo.predict(X))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Sem Diabetes', 'Com Diabetes'],
            yticklabels=['Sem Diabetes', 'Com Diabetes']
        )
        ax.set_title(f'Matriz de Confusão\n{nome}', fontsize=13)
        ax.set_ylabel('Real')
        ax.set_xlabel('Predito')

    plt.tight_layout()
    os.makedirs(pasta_saida, exist_ok=True)
    plt.savefig(f'{pasta_saida}/matriz_confusao.png', dpi=150)
    plt.show()
    print(f"[evaluate] Matriz de confusão salva em '{pasta_saida}/'")


def plotar_curva_roc(modelos_dict, X, y, pasta_saida='notebooks'):
    """Plota curva ROC para múltiplos modelos."""
    cores = ['steelblue', 'tomato', 'seagreen', 'darkorange']
    fig, ax = plt.subplots(figsize=(8, 6))

    for (nome, modelo), cor in zip(modelos_dict.items(), cores):
        y_prob = modelo.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc = roc_auc_score(y, y_prob)
        ax.plot(fpr, tpr, label=f'{nome} (AUC={auc:.3f})', color=cor, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC=0.500)')
    ax.set_xlabel('Taxa de Falsos Positivos', fontsize=12)
    ax.set_ylabel('Taxa de Verdadeiros Positivos (Recall)', fontsize=12)
    ax.set_title('Curva ROC', fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    os.makedirs(pasta_saida, exist_ok=True)
    plt.savefig(f'{pasta_saida}/curva_roc.png', dpi=150)
    plt.show()
    print(f"[evaluate] Curva ROC salva em '{pasta_saida}/'")


def comparar_modelos(resultados_lista, pasta_saida='notebooks'):
    """Gera gráfico comparativo entre modelos."""
    df = pd.DataFrame(resultados_lista).set_index('nome')
    metricas = ['accuracy', 'recall', 'f1', 'precision']
    x = np.arange(len(metricas))
    width = 0.8 / len(df)
    cores = ['steelblue', 'tomato', 'seagreen', 'darkorange']

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (nome, row) in enumerate(df.iterrows()):
        offset = (i - len(df) / 2 + 0.5) * width
        bars = ax.bar(x + offset, row[metricas], width, label=nome, color=cores[i], alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Recall ★', 'F1-score', 'Precision'], fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('Comparação de Modelos', fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    os.makedirs(pasta_saida, exist_ok=True)
    plt.savefig(f'{pasta_saida}/comparacao_modelos.png', dpi=150)
    plt.show()
    print(f"[evaluate] Gráfico comparativo salvo em '{pasta_saida}/'")


def pipeline_avaliacao(modelos_dict, X_test, y_test, pasta_saida='notebooks'):
    """Executa o pipeline completo de avaliação."""
    resultados = []
    for nome, modelo in modelos_dict.items():
        r = avaliar_modelo(nome, modelo, X_test, y_test)
        relatorio_completo(nome, modelo, X_test, y_test)
        resultados.append(r)

    plotar_matriz_confusao(modelos_dict, X_test, y_test, pasta_saida)
    plotar_curva_roc(modelos_dict, X_test, y_test, pasta_saida)
    comparar_modelos(resultados, pasta_saida)

    melhor = max(resultados, key=lambda r: r['recall'])
    print(f"\n[evaluate] Melhor modelo por Recall: {melhor['nome']} (Recall={melhor['recall']:.4f})")
    return resultados
