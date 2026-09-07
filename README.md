# Wine Quality - Exploratory Data Analysis (Homework 1)

**Disciplina:** Inteligência Computacional Aplicada
**Projeto:** Análise Exploratória de Dados - Wine Quality (UCI)
**Data de Entrega:** 18 de Setembro de 2026

---

## 1. Descrição do Projeto

Este repositório contém a implementação completa da análise exploratória (EDA) do conjunto de dados *Wine Quality* do UCI Machine Learning Repository.

O trabalho atende integralmente os 5 itens solicitados no enunciado do HW1:

1. **Descrição do Dataset**: Número de observações, preditores, classes e distribuição de frequência.
2. **Análise Univariada Incondicional**: Histogramas, box-plots, média, desvio padrão e assimetria para cada um dos 11 preditores (usando todos os dados).
3. **Análise Univariada Condicional à Classe**: As mesmas estatísticas e gráficos, mas calculadas separadamente para as classes `baixa`, `media` e `alta` (discretização da nota sensorial `quality`).
4. **Análise Bivariada**: Scatter plots coloridos por classe para todos os 55 pares de preditores e Matriz de Correlação de Pearson (com heatmap).
5. **Análise Multivariada (PCA)**: Implementação **do zero** (sem `sklearn.decomposition.PCA`) da Análise de Componentes Principais, com padronização dos dados, autodecomposição via `numpy.linalg.eigh` e projeção bidimensional colorida por classe.

**Diferencial:** Todos os passos são executados de forma **independente** para o vinho tinto (`N = 1599`) e para o vinho branco (`N = 4898`), permitindo a comparação direta das estruturas químico-físicas entre os dois tipos de vinho.

---

## 2. Autores e Contribuições

| Autor | Contribuição Principal |
| :--- | :--- |
| **Iuri Fernandes** | Carregamento e pré-processamento dos dados. Implementação das funções de estatística univariada (média, desvio, assimetria) e da lógica de discretização das classes. |
| **Jonas Albuquerque** | Pipeline de visualização (histogramas, box-plots, matriz de scatter plots bivariados e heatmap de correlação). Ajuste de estilos e geração das figuras para o relatório. |
| **João Otávio Brasil** | Implementação **manual** da PCA (do zero). Escrita da fundamentação teórica nos métodos do artigo e interpretação dos *loadings* e variância explicada. |

> **Nota:** Todos os autores participaram da revisão crítica dos resultados, discussão das implicações físico-químicas (comparação tinto vs branco) e da redação final do artigo IEEE.

---

## 3. Dependências e Instalação

O script foi escrito em Python 3.8+ e requer as seguintes bibliotecas:

- `numpy (>=1.21.0)`
- `pandas (>=1.3.0)`
- `scipy (>=1.7.0)`
- `matplotlib (>=3.4.0)`

Para instalar todas as dependências de uma vez, execute no terminal:

```bash
pip install numpy pandas scipy matplotlib
