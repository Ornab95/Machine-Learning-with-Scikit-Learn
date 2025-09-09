"""
Clustering Examples with Scikit-Learn

This script demonstrates various clustering algorithms:
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Model

Author: Machine Learning Tutorial Series
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample clustering datasets."""
    print("=== Creating Sample Clustering Data ===")
    
    # Generate synthetic dataset with clear clusters
    X_blobs, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, 
                                 random_state=42)
    
    print(f"Dataset shape: {X_blobs.shape}")
    print(f"True number of clusters: {len(np.unique(y_true))}")
    
    # Visualize the data
    plt.figure(figsize=(10, 8))
    plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.title('Sample Data with True Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('clustering_sample_data.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return X_blobs, y_true

def kmeans_clustering_example(X, y_true):
    """Demonstrate K-Means clustering."""
    print("\n=== K-Means Clustering ===")
    
    # Create and fit K-Means model
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)
    
    # Evaluate clustering
    silhouette = silhouette_score(X, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Original clusters
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.title('True Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # K-Means clusters
    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.title(f'K-Means Clusters (k={n_clusters})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # Cluster centers
    plt.subplot(1, 3, 3)
    centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=['Feature 1', 'Feature 2'])
    centers_df.index.name = 'Cluster'
    sns.heatmap(centers_df, annot=True, cmap='coolwarm', center=0)
    plt.title('Cluster Centers')
    
    plt.tight_layout()
    plt.savefig('kmeans_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return kmeans, y_pred, silhouette

def find_optimal_k(X, max_k=10):
    """Find optimal number of clusters using elbow method and silhouette analysis."""
    print("\n=== Finding Optimal Number of Clusters ===")
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, y_pred))
    
    # Plot elbow curve and silhouette scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Elbow method
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette scores
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimal_k_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find optimal k
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
    
    return optimal_k, silhouette_scores

def hierarchical_clustering_example(X, y_true):
    """Demonstrate Hierarchical clustering."""
    print("\n=== Hierarchical Clustering ===")
    
    # Create and fit Hierarchical clustering model
    n_clusters = 4
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    y_pred = hierarchical.fit_predict(X)
    
    # Evaluate clustering
    silhouette = silhouette_score(X, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Linkage method: {hierarchical.linkage}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    
    # Original clusters
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.title('True Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Hierarchical clusters
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
    plt.title(f'Hierarchical Clusters (n={n_clusters})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('hierarchical_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return hierarchical, y_pred, silhouette

def dbscan_clustering_example(X, y_true):
    """Demonstrate DBSCAN clustering."""
    print("\n=== DBSCAN Clustering ===")
    
    # Create and fit DBSCAN model
    dbscan = DBSCAN(eps=0.8, min_samples=5)
    y_pred = dbscan.fit_predict(X)
    
    # Count clusters (excluding noise points)
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    n_noise = list(y_pred).count(-1)
    
    print(f"Number of clusters found: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    print(f"Eps: {dbscan.eps}")
    print(f"Min samples: {dbscan.min_samples}")
    
    # Evaluate clustering (excluding noise points)
    if n_clusters > 1:
        # Remove noise points for evaluation
        mask = y_pred != -1
        if mask.sum() > 0:
            silhouette = silhouette_score(X[mask], y_pred[mask])
            ari = adjusted_rand_score(y_true[mask], y_pred[mask])
            print(f"Silhouette Score (excluding noise): {silhouette:.4f}")
            print(f"Adjusted Rand Index (excluding noise): {ari:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    
    # Original clusters
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.title('True Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # DBSCAN clusters
    plt.subplot(1, 2, 2)
    unique_labels = set(y_pred)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise
            col = [0, 0, 0, 1]
        
        class_member_mask = (y_pred == k)
        xy = X[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], alpha=0.7, 
                   s=60 if k != -1 else 20, label=f'Cluster {k}' if k != -1 else 'Noise')
    
    plt.title(f'DBSCAN Clusters (n={n_clusters}, noise={n_noise})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('dbscan_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return dbscan, y_pred, n_clusters

def gaussian_mixture_example(X, y_true):
    """Demonstrate Gaussian Mixture Model clustering."""
    print("\n=== Gaussian Mixture Model ===")
    
    # Create and fit GMM model
    n_components = 4
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    y_pred = gmm.predict(X)
    
    # Get probabilities
    y_proba = gmm.predict_proba(X)
    
    # Evaluate clustering
    silhouette = silhouette_score(X, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    print(f"Number of components: {n_components}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Log-likelihood: {gmm.score(X):.2f}")
    print(f"AIC: {gmm.aic(X):.2f}")
    print(f"BIC: {gmm.bic(X):.2f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Original clusters
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.title('True Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # GMM clusters
    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
    plt.title(f'GMM Clusters (n={n_components})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Probability heatmap for first cluster
    plt.subplot(1, 3, 3)
    plt.scatter(X[:, 0], X[:, 1], c=y_proba[:, 0], cmap='Reds', alpha=0.7)
    plt.colorbar()
    plt.title('Probability of Belonging to Cluster 0')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('gmm_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return gmm, y_pred, silhouette

def compare_clustering_algorithms(X, y_true):
    """Compare all clustering algorithms."""
    print("\n=== Clustering Algorithm Comparison ===")
    
    # Define algorithms
    algorithms = {
        'K-Means': KMeans(n_clusters=4, random_state=42, n_init=10),
        'Hierarchical': AgglomerativeClustering(n_clusters=4, linkage='ward'),
        'DBSCAN': DBSCAN(eps=0.8, min_samples=5),
        'GMM': GaussianMixture(n_components=4, random_state=42)
    }
    
    results = {}
    
    plt.figure(figsize=(20, 15))
    
    for i, (name, algorithm) in enumerate(algorithms.items()):
        # Fit the algorithm
        if name == 'GMM':
            algorithm.fit(X)
            y_pred = algorithm.predict(X)
        else:
            y_pred = algorithm.fit_predict(X)
        
        # Calculate metrics
        if name == 'DBSCAN':
            # Handle noise points for DBSCAN
            n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
            if n_clusters > 1:
                mask = y_pred != -1
                if mask.sum() > 1:
                    silhouette = silhouette_score(X[mask], y_pred[mask])
                    ari = adjusted_rand_score(y_true[mask], y_pred[mask])
                else:
                    silhouette = -1
                    ari = -1
            else:
                silhouette = -1
                ari = -1
        else:
            silhouette = silhouette_score(X, y_pred)
            ari = adjusted_rand_score(y_true, y_pred)
        
        results[name] = {'Silhouette': silhouette, 'ARI': ari}
        
        # Plot results
        plt.subplot(3, 4, i*3 + 1)
        plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
        plt.title('True Clusters')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        plt.subplot(3, 4, i*3 + 2)
        if name == 'DBSCAN':
            unique_labels = set(y_pred)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    col = [0, 0, 0, 1]
                class_member_mask = (y_pred == k)
                xy = X[class_member_mask]
                plt.scatter(xy[:, 0], xy[:, 1], c=[col], alpha=0.7, 
                           s=60 if k != -1 else 20)
        else:
            plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
        
        plt.title(f'{name} Clusters')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        # Add text with metrics
        plt.subplot(3, 4, i*3 + 3)
        plt.text(0.1, 0.8, f'Algorithm: {name}', fontsize=12, weight='bold')
        plt.text(0.1, 0.6, f'Silhouette: {silhouette:.3f}', fontsize=10)
        plt.text(0.1, 0.4, f'ARI: {ari:.3f}', fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('clustering_algorithms_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def iris_clustering_example():
    """Real-world example using Iris dataset."""
    print("\n=== Real-world Example: Iris Dataset ===")
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y_true = iris.target
    
    # Use PCA for visualization (reduce to 2D)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"Original dataset shape: {X.shape}")
    print(f"PCA dataset shape: {X_pca.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)  # Use original features
    
    # Evaluate
    silhouette = silhouette_score(X, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    
    # Plot results using PCA coordinates
    plt.figure(figsize=(15, 5))
    
    # True species
    plt.subplot(1, 3, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.title('True Species')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    # K-Means clusters
    plt.subplot(1, 3, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
    plt.title('K-Means Clusters')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    # Feature importance
    plt.subplot(1, 3, 3)
    feature_names = iris.feature_names
    plt.barh(feature_names, np.abs(pca.components_[0]), alpha=0.7, label='PC1')
    plt.barh(feature_names, np.abs(pca.components_[1]), alpha=0.7, label='PC2')
    plt.title('PCA Feature Importance')
    plt.xlabel('Absolute Component Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('iris_clustering_example.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run all clustering examples."""
    print("🎯 Machine Learning Clustering Tutorial")
    print("=" * 50)
    
    # Create sample data
    X, y_true = create_sample_data()
    
    # Find optimal number of clusters
    optimal_k, silhouette_scores = find_optimal_k(X)
    
    # Run individual clustering examples
    kmeans_model, kmeans_pred, kmeans_silhouette = kmeans_clustering_example(X, y_true)
    hierarchical_model, hierarchical_pred, hierarchical_silhouette = hierarchical_clustering_example(X, y_true)
    dbscan_model, dbscan_pred, dbscan_clusters = dbscan_clustering_example(X, y_true)
    gmm_model, gmm_pred, gmm_silhouette = gaussian_mixture_example(X, y_true)
    
    # Compare all algorithms
    results = compare_clustering_algorithms(X, y_true)
    
    # Real-world example
    iris_clustering_example()
    
    # Summary
    print("\n📊 Results Summary:")
    for algorithm, metrics in results.items():
        print(f"{algorithm}: Silhouette = {metrics['Silhouette']:.4f}, ARI = {metrics['ARI']:.4f}")
    
    print("\n✅ Clustering tutorial completed!")
    print("\nKey takeaways:")
    print("1. K-Means works well for spherical clusters")
    print("2. Hierarchical clustering doesn't require pre-specifying number of clusters")
    print("3. DBSCAN can find clusters of arbitrary shape and identify noise")
    print("4. GMM provides probabilistic cluster assignments")
    print("5. Use silhouette score and ARI to evaluate clustering quality")
    print("6. Consider the nature of your data when choosing algorithms")

if __name__ == "__main__":
    main()