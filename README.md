# Wholesale Customers Clustering

[![Python](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![OpenML](https://img.shields.io/badge/OpenML-dataset-yellow)](https://www.openml.org/)
[![Status](https://img.shields.io/badge/status-active-brightgreen)]()

---

## 📌 Descripción

Proyecto de **clustering de clientes mayoristas** utilizando el dataset [Wholesale Customers](https://www.openml.org/d/4534) de OpenML.  
Este proyecto implementa un **pipeline completo** de análisis, incluyendo preprocesamiento, determinación de clusters óptimos, clustering con **K-Means** y **DBSCAN**, análisis estadístico por cluster y visualización mediante **PCA**.

Ideal para portafolio académico o profesional, demostrando buenas prácticas en **segmentación de clientes y análisis exploratorio multivariante**.

---

## 🛠 Características

- **Preprocesamiento de datos**:
  - Escalado de características con `StandardScaler`
  - Limpieza y verificación de datos

- **Determinación de clusters óptimos**:
  - Método del Codo (`Elbow Method`)
  - Silhouette Score

- **Clustering**:
  - **K-Means**: segmentación basada en centroides
  - **DBSCAN**: detección de clusters y outliers por densidad

- **Análisis estadístico por cluster**:
  - Estadísticas descriptivas
  - Matriz de correlación por cluster

- **Visualización**:
  - Reducción de dimensionalidad a 2D mediante **PCA**
  - Gráficos de dispersión y heatmaps

- **Exportación de resultados**:
  - Archivo `wholesale_clusters.csv` con la asignación de clusters

---

## 📊 Resultados

- Segmentación clara de clientes mayoristas según patrones de consumo  
- Identificación de correlaciones entre variables dentro de cada cluster  
- Visualizaciones 2D que facilitan la interpretación de los clusters  

*(Se recomienda agregar imágenes o GIF de los gráficos generados en la carpeta `figures/` para mayor impacto visual.)*

---

## ⚡ Instalación

Instala las librerías necesarias:

```bash
pip install openml pandas numpy matplotlib seaborn scikit-learn
