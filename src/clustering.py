import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import K_RANGE, RANDOM_STATE, PCA_COMPONENTS

def cluster_pure(df, features):
    """Cluster using only base statistics (NO target leakage)."""
    logger.info("[Pure] Computing clusters without engineered features...")
    
    X_raw = df[features].apply(pd.to_numeric, errors='coerce')
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_filled = imputer.fit_transform(X_raw)
    X_scaled = scaler.fit_transform(X_filled)
    
    # Find best k using silhouette score
    best_k = None
    best_score = -1
    sil_scores = []
    
    for k in K_RANGE:
        kmeans_temp = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans_temp.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        sil_scores.append(score)
        
        if score > best_score:
            best_score = score
            best_k = k
        
        logger.info(f"  k={k}: silhouette={score:.3f}")
    
    # Train final model
    kmeans_pure = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    df['cluster_km_pure'] = kmeans_pure.fit_predict(X_scaled)
    
    logger.info(f"[Pure] Best k={best_k}, silhouette={best_score:.3f}")
    
    # Visualize
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=df['cluster_km_pure'], palette='tab10', s=50)
    plt.title('K-Means Pure Clusters (PCA 2D)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.tight_layout()
    plt.savefig('results/clusters_pure.png', dpi=100)
    plt.close()
    
    return df, best_k, best_score

def cluster_engineered(df, features):
    """Cluster including goal variables (FOR COMPARISON ONLY - shows leakage)."""
    logger.info("[Engineered] Computing clusters with goal variables...")
    
    features_eng = features + ['FT_total_goals', 'HT_total_goals', 'SHT_total_goals']
    X_raw = df[features_eng].apply(pd.to_numeric, errors='coerce')
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_filled = imputer.fit_transform(X_raw)
    X_scaled = scaler.fit_transform(X_filled)
    
    # Find best k
    best_k = None
    best_score = -1
    
    for k in K_RANGE:
        kmeans_temp = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans_temp.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        
        if score > best_score:
            best_score = score
            best_k = k
        
        logger.info(f"  k={k}: silhouette={score:.3f}")
    
    # Train final model
    kmeans_eng = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    df['cluster_km_engineered'] = kmeans_eng.fit_predict(X_scaled)
    
    logger.info(f"[Engineered] Best k={best_k}, silhouette={best_score:.3f}")
    
    # Visualize
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=df['cluster_km_engineered'], palette='tab10', s=50)
    plt.title('K-Means Engineered Clusters (PCA 2D)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.tight_layout()
    plt.savefig('results/clusters_engineered.png', dpi=100)
    plt.close()
    
    return df, best_k, best_score
