import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. Carregar os dados
# Substitua os caminhos dos arquivos pelos seus próprios
x_train = pd.read_csv('train/X_train.txt', delim_whitespace=True, header=None)
y_train = pd.read_csv('train/y_train.txt', delim_whitespace=True, header=None, names=['activity'])
subject_train = pd.read_csv('train/subject_train.txt', delim_whitespace=True, header=None, names=['subject'])

# 2. Normalização dos dados (padronização)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# 3. Redução de dimensionalidade usando PCA
pca = PCA()
x_train_pca = pca.fit_transform(x_train_scaled)

# Variância explicada acumulada
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Determinar o número de componentes principais para >90% da variância
n_components = np.argmax(explained_variance_ratio >= 0.9) + 1
x_train_pca_reduced = x_train_pca[:, :n_components]

# Plot da variância explicada acumulada
plt.figure(figsize=(10, 6))
plt.plot(explained_variance_ratio, marker='o', linestyle='--', color='b')
plt.axhline(y=0.9, color='r', linestyle='-')
plt.axvline(x=n_components - 1, color='r', linestyle='--', label=f'{n_components} componentes')
plt.title('Variância Explicada Acumulada por PCA')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Explicada Acumulada')
plt.legend()
plt.grid()
plt.show()

# 4. Método do Cotovelo e Silhouette Score
inertia = []
silhouette_scores = []
cluster_range = range(2, 11)  # Testar valores de K de 2 a 10

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10, max_iter=300)
    kmeans.fit(x_train_pca_reduced)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(x_train_pca_reduced, kmeans.labels_))

# Gráficos: Inércia (Elbow) e Silhouette Score
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico do cotovelo
ax[0].plot(cluster_range, inertia, marker='o', linestyle='--', color='b')
ax[0].set_title('Método do Cotovelo')
ax[0].set_xlabel('Número de Clusters (K)')
ax[0].set_ylabel('Inércia')
ax[0].grid()

# Gráfico do silhouette score
ax[1].plot(cluster_range, silhouette_scores, marker='o', linestyle='--', color='g')
ax[1].set_title('Silhouette Score')
ax[1].set_xlabel('Número de Clusters (K)')
ax[1].set_ylabel('Silhouette Score')
ax[1].grid()

plt.tight_layout()
plt.show()

# Melhor número de clusters baseado no silhouette score
best_k = cluster_range[np.argmax(silhouette_scores)]
print(f"Melhor número de clusters (K) baseado no Silhouette Score: {best_k}")

# 5. Treinar K-means com o melhor número de clusters
kmeans_final = KMeans(n_clusters=best_k, init='k-means++', random_state=42, n_init=10, max_iter=300)
kmeans_final.fit(x_train_pca_reduced)

# Atribuir clusters às amostras
clusters = kmeans_final.labels_

# 6. Visualização dos clusters em 2D (usando PCA com 2 componentes para visualização)
pca_2d = PCA(n_components=2)
x_train_2d = pca_2d.fit_transform(x_train_scaled)

plt.figure(figsize=(10, 8))
plt.scatter(x_train_2d[:, 0], x_train_2d[:, 1], c=clusters, cmap='viridis', s=10)
plt.title('Clusters Formados pelo K-means (Reduzido para 2D)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(label='Cluster')
plt.grid()
plt.show()
