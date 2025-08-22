# librerias
import openml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

#Descarga el dataset
def get_and_preprocess_data():
    dataset = openml.datasets.get_dataset(4534)
    X, y, _, attr = dataset.get_data(dataset_format="dataframe")
    df = X.copy()
    print('Primeros registros del dataset:')
    print(df.head())

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df, df_scaled

#Determinar el número óptimo de clusters 
def find_optimal_clusters(df_scaled):
    inertia = []
    sil_scores = []
    K = range(2, 10)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df_scaled)
        inertia.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(df_scaled, labels))

    #Gráfica del Método del Codo
    plt.figure(figsize=(8, 5))
    plt.plot(K, inertia, 'bo-')
    plt.xlabel('Número de clusters')
    plt.ylabel('Inertia')
    plt.title('Método del Codo')
    plt.grid() 
    plt.show()

    #Muestra los Silhouette Scores
    for k, score in zip(K, sil_scores):
        print(f'k={k} -> Silhouette Score={score:.3f}')
    
    k_optimo = K[sil_scores.index(max(sil_scores))]
    print(f'\nNúmero óptimo de clusters según Silhouette: {k_optimo}')
    return k_optimo

#Aplica K-Means y DBSCAN 
def perform_clustering_and_analysis(df, df_scaled, k_clusters):
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    df['Cluster_KMeans'] = kmeans.fit_predict(df_scaled)
    
    # Análisis de variables por cluster
    cluster_summary = df.select_dtypes(include=['number']).groupby('Cluster_KMeans').mean()
    print('\nResumen de variables por clúster (K-Means):')
    print(cluster_summary)
    
    # Análisis de distribución por cluster
    cluster_stats = df.groupby('Cluster_KMeans').describe()
    print('\nEstadísticas de distribución por clúster (K-Means):')
    print(cluster_stats)

    # Evaluación con métricas adicionales
    print('\nMétricas de evaluación (K-Means):')
    print('Calinski-Harabasz:', calinski_harabasz_score(df_scaled, df['Cluster_KMeans']))
    print('Davies-Bouldin:', davies_bouldin_score(df_scaled, df['Cluster_KMeans']))

    # Probar otro algoritmo de clustering (DBSCAN)
    dbscan = DBSCAN(eps=2, min_samples=5)
    df['Cluster_DBSCAN'] = dbscan.fit_predict(df_scaled)
    print("\nResultados de DBSCAN agregados al DataFrame.")

    return df

#Visualiza los clusters
def visualize_clusters(df, df_scaled):
    # Aplicar PCA una sola vez para la visualización
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)
    df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]
    
    # Visualización de K-Means con PCA
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster_KMeans', palette='Set2', s=100)
    plt.title("Visualización de Clusters K-Means con PCA")
    plt.grid() 
    plt.show()

    # Matriz de correlación por cluster (corregido)
    for c in df['Cluster_KMeans'].unique():
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[df['Cluster_KMeans'] == c].select_dtypes(include=['number']).corr(), cmap="coolwarm", annot=True, fmt=".2f")
        plt.title(f'Matriz de correlación para Cluster {c}')
        plt.show()



# Función principal
def main():
    df, df_scaled = get_and_preprocess_data()
    k_optimo = find_optimal_clusters(df_scaled)
    df = perform_clustering_and_analysis(df, df_scaled, k_optimo)
    visualize_clusters(df, df_scaled)
    
    # Guardar resultados
    df.to_csv("wholesale_clusters.csv", index=False)
    print("\nArchivo 'wholesale_clusters.csv' guardado con clusters.")

if __name__ == '__main__':
    main()