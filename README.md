# Definindo o conteúdo do README.md
readme_content = """
# Relatório Técnico: Implementa-o-e-An-lise-do-Algoritmo-de-K-means-com-o-Dataset-Human-Activity-Recognition

**Nome do Residente:** Leandro de Oliveira e Sumaia Suzart Argôlo Nunesmaia  
**Data de Entrega:** 27/11/2024  
**RESTIC-36 - CEPEDI**  

## Resumo

Este projeto tem como objetivo aplicar o algoritmo de K-means para o reconhecimento de atividades humanas com base em um conjunto de dados de sensores. O script `main.py` implementa a técnica de clustering não supervisionado para identificar padrões nas atividades de um grupo de pessoas. São realizadas análises para determinar o número ideal de clusters utilizando os métodos do cotovelo e o silhouette score. Além disso, são aplicadas técnicas de normalização para garantir que as variáveis influenciem o modelo de forma equilibrada. O modelo final é validado com métricas de avaliação e visualizações para entender a coesão e separação dos clusters formados.

## Introdução

Com o avanço da tecnologia de sensores, o reconhecimento de atividades humanas se tornou uma área importante de estudo em diversas aplicações, como saúde, segurança e monitoramento. Este projeto utiliza o algoritmo K-means para identificar padrões de comportamento a partir de dados de sensores, com o objetivo de classificar as atividades realizadas por um indivíduo. O K-means é uma técnica de clustering que divide os dados em grupos com base em suas características semelhantes.

### Conjunto de Dados
LINK DOS DADOS:https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

O conjunto de dados utilizado contém informações sobre as atividades humanas registradas por sensores. As variáveis mais relevantes incluem:
- **X, Y, Z (acelerômetro):** Dados de aceleração nas direções X, Y e Z.
- **atividade:** A classe de atividade que o indivíduo está realizando (por exemplo, caminhada, corrida, descanso, etc.).
- **tempo:** Um timestamp ou índice relacionado à sequência de dados.

O conjunto de dados pode ter múltiplas variáveis que capturam as atividades em diferentes dimensões. Para garantir um bom desempenho do algoritmo, é necessário aplicar técnicas de pré-processamento, como normalização, e técnicas de redução de dimensionalidade, se necessário.

## Metodologia

### Análise Exploratória de Dados (AED)

A análise exploratória foi conduzida para entender a distribuição dos dados e suas possíveis correlações. Gráficos de dispersão e histogramas foram gerados para examinar as relações entre as variáveis. Mapas de calor de correlação ajudaram a identificar variáveis com maior influência sobre a formação dos clusters.

### Implementação do Algoritmo

O algoritmo K-means foi implementado utilizando a biblioteca Scikit-Learn. Para garantir uma inicialização eficiente dos centróides, utilizamos o método K-means++, que ajuda a reduzir o risco de convergência para um mínimo local.

Antes de aplicar o algoritmo, os dados foram normalizados usando o `StandardScaler`, garantindo que todas as variáveis contribuam igualmente para a formação dos clusters. A escolha do número de clusters, K, foi realizada com base na análise do método do cotovelo e no cálculo do silhouette score.

### Validação e Ajuste de Hiperparâmetros

Para garantir a robustez e a qualidade do modelo, foi utilizado o método do cotovelo para determinar o número ideal de clusters. Além disso, o silhouette score foi calculado para avaliar a coesão e a separação dos clusters formados, ajudando a identificar a qualidade dos agrupamentos.

## Resultados

### Métricas de Avaliação

Após a execução do algoritmo K-means, as seguintes métricas foram obtidas:

- **Soma das Distâncias ao Quadrado (Inércia):** Mede o quão bem os pontos estão agrupados ao redor dos centróides.
- **Silhouette Score:** Avalia a coesão e separação dos clusters, com valores próximos de 1 indicando bons clusters.
- **Número de Clusters (K):** O valor ideal de K foi selecionado com base na análise do cotovelo e do silhouette score.

### Visualizações

- **Método do Cotovelo:** Gráfico que mostra a relação entre o número de clusters e a soma das distâncias ao quadrado (inércia), para identificar o ponto de inflexão.
- **Silhouette Score:** Gráfico que ilustra a qualidade dos clusters, permitindo a análise da separação entre eles.
- **Distribuição dos Clusters:** Gráfico de dispersão dos clusters formados, com a redução de dimensionalidade utilizando PCA, se necessário.

```python
# Exemplo de visualização em Python
import matplotlib.pyplot as plt
import seaborn as sns

# Gráfico do Método do Cotovelo
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia_values, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
plt.show()

# Gráfico de Silhouette Score
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score por Número de Clusters')
plt.xlabel('Número de Clusters')
plt.ylabel('Silhouette Score')
plt.show()
