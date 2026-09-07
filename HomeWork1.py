"""
Wine Quality - Analise Exploratoria (Homework 1 - Inteligencia Computacional Aplicada)
========================================================================================

Este script realiza a analise exploratoria pedida no enunciado do HW1, para os
datasets de vinho tinto e vinho branco (UCI Wine Quality), tratados SEPARADAMENTE.

Para cada um dos dois vinhos, o script:
  1. Descreve o dataset (N observacoes, D preditores, L classes, distribuicao de classes)
  2. Analise mono-variada INCONDICIONAL: histogramas, box-plots, media, desvio padrao
     e skewness de cada um dos D preditores (usando todas as N observacoes)
  3. Analise mono-variada CONDICIONAL a classe: histogramas, box-plots, media, desvio
     padrao e skewness de cada preditor, para cada uma das L classes
  4. Analise bi-variada INCONDICIONAL: scatter plots entre pares de preditores
     (coloridos por classe) e matriz de correlacao de Pearson
  5. Analise multi-variada: PCA implementado do zero (sem sklearn.decomposition nem
     funcoes prontas de PCA), retendo as duas primeiras componentes principais

A "classe" (rotulo) usada e uma discretizacao da variavel `quality` (nota sensorial)
em L=3 faixas: baixa, media e alta. Essa escolha mantem o proposito original do
dataset (modelar a qualidade do vinho a partir de propriedades fisico-quimicas).

Dependencias: numpy, pandas, matplotlib, scipy (apenas para mean/std/skew e leitura
de CSV; a PCA e feita manualmente com numpy.linalg).

Os dados sao baixados automaticamente do UCI (necessita internet). Caso a maquina
nao tenha acesso a internet, baixe manualmente os arquivos:
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
e coloque-os na mesma pasta deste script (o script detecta os arquivos locais
automaticamente se o download falhar).
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # nao precisa de display para salvar as figuras em disco
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------
# CONFIGURACOES GERAIS
# --------------------------------------------------------------------------------

RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

RED_LOCAL = "winequality-red.csv"
WHITE_LOCAL = "winequality-white.csv"

OUTPUT_DIR = "outputs"

# Bordas das faixas de qualidade -> 3 classes: baixa, media, alta
# (intervalos fechados a direita: baixa = quality <= 4; media = 5-6; alta >= 7)
QUALITY_BINS = [0, 4, 6, 10]
QUALITY_LABELS = ["baixa", "media", "alta"]

PREDICTOR_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]

CLASS_COLORS = {"baixa": "#d62728", "media": "#1f77b4", "alta": "#2ca02c"}


# --------------------------------------------------------------------------------
# 0. CARREGAMENTO DOS DADOS
# --------------------------------------------------------------------------------

def load_wine_data(url, local_path):
    """Carrega o CSV do UCI (separador ';'). Tenta baixar da internet; se falhar,
    tenta ler um arquivo local com o mesmo nome na pasta atual."""
    try:
        df = pd.read_csv(url, sep=";")
        print(f"[ok] Dados baixados de {url}")
    except Exception as exc:
        if os.path.exists(local_path):
            df = pd.read_csv(local_path, sep=";")
            print(f"[ok] Dados lidos do arquivo local {local_path}")
        else:
            raise RuntimeError(
                f"Nao foi possivel baixar {url} nem encontrar {local_path}. "
                f"Baixe o arquivo manualmente e coloque-o nesta pasta."
            ) from exc
    return df


def add_class_labels(df):
    """Cria a coluna 'class' a partir da discretizacao de 'quality' em 3 faixas."""
    df = df.copy()
    df["class"] = pd.cut(
        df["quality"], bins=QUALITY_BINS, labels=QUALITY_LABELS, include_lowest=True
    )
    return df


# --------------------------------------------------------------------------------
# 1. DESCRICAO DO DATASET
# --------------------------------------------------------------------------------

def describe_dataset(df, wine_name):
    N = len(df)
    D = len(PREDICTOR_COLUMNS)
    dist = df["class"].value_counts().reindex(QUALITY_LABELS)
    L = dist.count()

    print(f"\n=== {wine_name.upper()} ===")
    print(f"N (observacoes) = {N}")
    print(f"D (preditores)  = {D}")
    print(f"L (classes)     = {L}")
    print("Distribuicao de classes:")
    for label, count in dist.items():
        pct = 100 * count / N
        print(f"   {label:6s}: {count:5d}  ({pct:5.1f}%)")

    return {"N": N, "D": D, "L": L, "distribution": dist}


# --------------------------------------------------------------------------------
# 2 e 3. ESTATISTICAS MONO-VARIADAS (INCONDICIONAL e CONDICIONAL)
# --------------------------------------------------------------------------------

def monovariate_stats(df, group_col=None):
    """Calcula media, desvio padrao e skewness de cada preditor.
    Se group_col for None -> analise incondicional (usa todas as N observacoes).
    Se group_col for 'class' -> analise condicional (uma linha por classe x preditor).
    """
    rows = []
    if group_col is None:
        for col in PREDICTOR_COLUMNS:
            x = df[col].values
            rows.append({
                "predictor": col,
                "mean": np.mean(x),
                "std": np.std(x, ddof=1),
                "skewness": stats.skew(x),
            })
        return pd.DataFrame(rows)
    else:
        for label in QUALITY_LABELS:
            subset = df[df[group_col] == label]
            for col in PREDICTOR_COLUMNS:
                x = subset[col].values
                rows.append({
                    "class": label,
                    "predictor": col,
                    "N_l": len(x),
                    "mean": np.mean(x) if len(x) > 0 else np.nan,
                    "std": np.std(x, ddof=1) if len(x) > 1 else np.nan,
                    "skewness": stats.skew(x) if len(x) > 2 else np.nan,
                })
        return pd.DataFrame(rows)


def plot_unconditional_histograms_boxplots(df, wine_name, out_dir):
    """Um histograma e um box-plot por preditor (incondicional)."""
    fig_dir = os.path.join(out_dir, wine_name, "unconditional")
    os.makedirs(fig_dir, exist_ok=True)

    for col in PREDICTOR_COLUMNS:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        axes[0].hist(df[col], bins=30, color="#4c72b0", edgecolor="white")
        axes[0].set_title(f"Histograma - {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frequencia")

        axes[1].boxplot(df[col], vert=True)
        axes[1].set_title(f"Box-plot - {col}")
        axes[1].set_xticklabels([col], rotation=20, ha="right")

        fig.tight_layout()
        safe_name = col.replace(" ", "_")
        fig.savefig(os.path.join(fig_dir, f"{safe_name}.png"), dpi=150)
        plt.close(fig)


def plot_conditional_histograms_boxplots(df, wine_name, out_dir):
    """Histograma sobreposto (por classe) e box-plot agrupado por classe, por preditor."""
    fig_dir = os.path.join(out_dir, wine_name, "conditional")
    os.makedirs(fig_dir, exist_ok=True)

    for col in PREDICTOR_COLUMNS:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

        # histogramas sobrepostos, um por classe
        for label in QUALITY_LABELS:
            data = df.loc[df["class"] == label, col]
            axes[0].hist(
                data, bins=25, alpha=0.5, label=label,
                color=CLASS_COLORS[label], density=True,
            )
        axes[0].set_title(f"Histograma condicional - {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Densidade")
        axes[0].legend(fontsize=7)

        # box-plot agrupado por classe
        data_by_class = [df.loc[df["class"] == label, col] for label in QUALITY_LABELS]
        bp = axes[1].boxplot(data_by_class, tick_labels=QUALITY_LABELS, patch_artist=True)
        for patch, label in zip(bp["boxes"], QUALITY_LABELS):
            patch.set_facecolor(CLASS_COLORS[label])
            patch.set_alpha(0.5)
        axes[1].set_title(f"Box-plot condicional - {col}")

        fig.tight_layout()
        safe_name = col.replace(" ", "_")
        fig.savefig(os.path.join(fig_dir, f"{safe_name}.png"), dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------------
# 4. ANALISE BI-VARIADA (scatter plots + matriz de correlacao)
# --------------------------------------------------------------------------------

def plot_scatter_matrix_sample(df, wine_name, out_dir, max_pairs=None):
    """Gera um scatter plot para cada par (di, dj) de preditores, colorido por classe.
    Com D=11 preditores ha C(11,2)=55 pares -- todos sao gerados e salvos em disco;
    no artigo, selecionem apenas os pares mais relevantes para exibir nas figuras."""
    fig_dir = os.path.join(out_dir, wine_name, "scatter_pairs")
    os.makedirs(fig_dir, exist_ok=True)

    pairs = [
        (PREDICTOR_COLUMNS[i], PREDICTOR_COLUMNS[j])
        for i in range(len(PREDICTOR_COLUMNS))
        for j in range(i + 1, len(PREDICTOR_COLUMNS))
    ]
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    for di, dj in pairs:
        fig, ax = plt.subplots(figsize=(4.2, 4))
        for label in QUALITY_LABELS:
            subset = df[df["class"] == label]
            ax.scatter(
                subset[di], subset[dj], s=8, alpha=0.5,
                color=CLASS_COLORS[label], label=label,
            )
        ax.set_xlabel(di)
        ax.set_ylabel(dj)
        ax.legend(fontsize=7)
        fig.tight_layout()
        safe_name = f"{di}_vs_{dj}".replace(" ", "_")
        fig.savefig(os.path.join(fig_dir, f"{safe_name}.png"), dpi=150)
        plt.close(fig)


def correlation_matrix(df, wine_name, out_dir):
    """Calcula e plota a matriz de correlacao de Pearson entre os D preditores."""
    fig_dir = os.path.join(out_dir, wine_name)
    os.makedirs(fig_dir, exist_ok=True)

    corr = df[PREDICTOR_COLUMNS].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(PREDICTOR_COLUMNS)))
    ax.set_yticks(range(len(PREDICTOR_COLUMNS)))
    ax.set_xticklabels(PREDICTOR_COLUMNS, rotation=90, fontsize=7)
    ax.set_yticklabels(PREDICTOR_COLUMNS, fontsize=7)
    fig.colorbar(im, ax=ax, label="Correlacao de Pearson")
    ax.set_title(f"Matriz de correlacao - {wine_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "correlation_matrix.png"), dpi=150)
    plt.close(fig)

    corr.to_csv(os.path.join(fig_dir, "correlation_matrix.csv"))
    return corr


# --------------------------------------------------------------------------------
# 5. PCA IMPLEMENTADO DO ZERO (sem sklearn.decomposition ou similares)
# --------------------------------------------------------------------------------

def pca_from_scratch(X, n_components=2):
    """Implementacao manual de PCA.

    Passos:
      1. Padronizar os dados (media 0, variancia 1 em cada coluna) -- necessario
         pois os preditores tem escalas muito diferentes (ex: 'density' ~ 0.99 a 1.00
         vs 'total sulfur dioxide' ~ dezenas a centenas).
      2. Calcular a matriz de covariancia (D x D) dos dados padronizados.
      3. Calcular autovalores e autovetores da matriz de covariancia
         (usa-se numpy.linalg.eigh pois a matriz de covariancia e simetrica).
      4. Ordenar os autovalores (e autovetores correspondentes) em ordem decrescente.
      5. Projetar os dados padronizados nos 'n_components' autovetores principais.

    Retorna: scores (N x n_components), variancia explicada por componente,
             variancia explicada acumulada, e os proprios autovetores (loadings).
    """
    # 1) padronizacao (z-score)
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    X_std = (X - mean) / std

    # 2) matriz de covariancia (D x D)
    N = X_std.shape[0]
    cov_matrix = (X_std.T @ X_std) / (N - 1)

    # 3) autovalores e autovetores (matriz simetrica -> eigh)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 4) ordenar em ordem decrescente de autovalor
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # 5) projetar nas n_components principais
    components = eigenvectors[:, :n_components]
    scores = X_std @ components

    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    return {
        "scores": scores,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained_variance_ratio,
        "loadings": components,
    }


def plot_pca(df, wine_name, out_dir):
    X = df[PREDICTOR_COLUMNS].values.astype(float)
    result = pca_from_scratch(X, n_components=2)
    scores = result["scores"]
    var_ratio = result["explained_variance_ratio"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for label in QUALITY_LABELS:
        mask = (df["class"] == label).values
        ax.scatter(
            scores[mask, 0], scores[mask, 1], s=10, alpha=0.55,
            color=CLASS_COLORS[label], label=label,
        )
    ax.set_xlabel(f"PC1 ({100*var_ratio[0]:.1f}% da variancia)")
    ax.set_ylabel(f"PC2 ({100*var_ratio[1]:.1f}% da variancia)")
    ax.set_title(f"PCA (2 componentes) - {wine_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()

    fig_dir = os.path.join(out_dir, wine_name)
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "pca_scatter.png"), dpi=150)
    plt.close(fig)

    # tabela de loadings (contribuicao de cada preditor para PC1 e PC2)
    loadings_df = pd.DataFrame(
        result["loadings"], index=PREDICTOR_COLUMNS, columns=["PC1", "PC2"]
    )
    loadings_df.to_csv(os.path.join(fig_dir, "pca_loadings.csv"))

    print(f"\n[{wine_name}] Variancia explicada: "
          f"PC1={100*var_ratio[0]:.1f}%  PC2={100*var_ratio[1]:.1f}%  "
          f"(acumulada={100*(var_ratio[0]+var_ratio[1]):.1f}%)")

    return result


# --------------------------------------------------------------------------------
# PIPELINE COMPLETO PARA UM VINHO
# --------------------------------------------------------------------------------

def run_full_analysis(url, local_path, wine_name, out_dir):
    df = load_wine_data(url, local_path)
    df = add_class_labels(df)

    describe_dataset(df, wine_name)

    uncond_stats = monovariate_stats(df, group_col=None)
    cond_stats = monovariate_stats(df, group_col="class")

    wine_out = os.path.join(out_dir, wine_name)
    os.makedirs(wine_out, exist_ok=True)
    uncond_stats.to_csv(os.path.join(wine_out, "unconditional_stats.csv"), index=False)
    cond_stats.to_csv(os.path.join(wine_out, "conditional_stats.csv"), index=False)

    plot_unconditional_histograms_boxplots(df, wine_name, out_dir)
    plot_conditional_histograms_boxplots(df, wine_name, out_dir)

    plot_scatter_matrix_sample(df, wine_name, out_dir)  # gera todos os 55 pares
    corr = correlation_matrix(df, wine_name, out_dir)

    pca_result = plot_pca(df, wine_name, out_dir)

    return {
        "df": df,
        "unconditional_stats": uncond_stats,
        "conditional_stats": cond_stats,
        "correlation": corr,
        "pca": pca_result,
    }


# --------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Rodando analise completa para VINHO TINTO...")
    red_results = run_full_analysis(RED_URL, RED_LOCAL, "red", OUTPUT_DIR)

    print("\nRodando analise completa para VINHO BRANCO...")
    white_results = run_full_analysis(WHITE_URL, WHITE_LOCAL, "white", OUTPUT_DIR)

    print(f"\nConcluido. Resultados salvos em: {os.path.abspath(OUTPUT_DIR)}")
    print("Estrutura de pastas:")
    print("  outputs/red/...     -> tabelas e figuras do vinho tinto")
    print("  outputs/white/...   -> tabelas e figuras do vinho branco")


if __name__ == "__main__":
    main()