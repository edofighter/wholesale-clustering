# Wholesale Customers Clustering

Clustering de clientes mayoristas usando K-Means y DBSCAN con análisis de variables y visualización con PCA.

## Descripción
Este proyecto descarga el dataset **Wholesale Customers** desde OpenML y realiza:

- Escalado de datos con `StandardScaler`.
- Determinación del número óptimo de clusters usando el **Método del Codo** y **Silhouette Score**.
- Aplicación de **K-Means** y **DBSCAN**.
- Análisis estadístico por cluster.
- Visualización de clusters reducidos a 2D con **PCA**.
- Matriz de correlación por cluster.

El resultado final se guarda como `wholesale_clusters.csv`.

## Requisitos
- Python 3.8+
- Librerías:
```bash
pip install openml pandas matplotlib seaborn scikit-learn
