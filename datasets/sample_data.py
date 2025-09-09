"""
Sample datasets for Machine Learning practice

This module provides easy access to sample datasets for learning and practicing
machine learning concepts.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, make_blobs

def create_simple_classification_data(n_samples=200, n_features=2, n_classes=3, random_state=42):
    """
    Create a simple classification dataset for beginners.
    
    Parameters:
    -----------
    n_samples : int, default=200
        Number of samples to generate
    n_features : int, default=2
        Number of features
    n_classes : int, default=3
        Number of classes
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    X : numpy array, shape (n_samples, n_features)
        Features
    y : numpy array, shape (n_samples,)
        Target labels
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_informative=n_features,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state
    )
    return X, y

def create_simple_regression_data(n_samples=200, n_features=1, noise=15, random_state=42):
    """
    Create a simple regression dataset for beginners.
    
    Parameters:
    -----------
    n_samples : int, default=200
        Number of samples to generate
    n_features : int, default=1
        Number of features
    noise : float, default=15
        Standard deviation of gaussian noise
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    X : numpy array, shape (n_samples, n_features)
        Features
    y : numpy array, shape (n_samples,)
        Target values
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    return X, y

def create_clustering_data(n_samples=300, n_centers=4, cluster_std=1.0, random_state=42):
    """
    Create a clustering dataset for unsupervised learning practice.
    
    Parameters:
    -----------
    n_samples : int, default=300
        Number of samples to generate
    n_centers : int, default=4
        Number of cluster centers
    cluster_std : float, default=1.0
        Standard deviation of clusters
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    X : numpy array, shape (n_samples, 2)
        Features (2D for easy visualization)
    y : numpy array, shape (n_samples,)
        True cluster labels
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=random_state
    )
    return X, y

def save_sample_datasets():
    """Create and save sample datasets as CSV files."""
    
    # Classification dataset
    X_class, y_class = create_simple_classification_data()
    class_df = pd.DataFrame(X_class, columns=['feature_1', 'feature_2'])
    class_df['target'] = y_class
    class_df.to_csv('classification_dataset.csv', index=False)
    
    # Regression dataset
    X_reg, y_reg = create_simple_regression_data()
    reg_df = pd.DataFrame(X_reg, columns=['feature'])
    reg_df['target'] = y_reg
    reg_df.to_csv('regression_dataset.csv', index=False)
    
    # Clustering dataset
    X_cluster, y_cluster = create_clustering_data()
    cluster_df = pd.DataFrame(X_cluster, columns=['feature_1', 'feature_2'])
    cluster_df['true_cluster'] = y_cluster
    cluster_df.to_csv('clustering_dataset.csv', index=False)
    
    print("Sample datasets created and saved:")
    print("- classification_dataset.csv")
    print("- regression_dataset.csv") 
    print("- clustering_dataset.csv")

if __name__ == "__main__":
    save_sample_datasets()