"""
Data Preprocessing Basics with Scikit-Learn

This script demonstrates fundamental data preprocessing techniques:
- Loading and exploring data
- Handling missing values
- Feature scaling and normalization
- Encoding categorical variables

Author: Machine Learning Tutorial Series
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris, fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns

def load_sample_data():
    """Load and return sample data for preprocessing examples."""
    print("=== Loading Sample Data ===")
    
    # Load Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target
    
    print(f"Iris dataset shape: {iris_df.shape}")
    print("\nFirst 5 rows:")
    print(iris_df.head())
    
    return iris_df

def explore_data(df):
    """Explore the basic properties of the dataset."""
    print("\n=== Data Exploration ===")
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    
    # Check for missing values
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Basic statistics
    print(f"\nBasic statistics:")
    print(df.describe())
    
    return df

def handle_missing_values(df):
    """Demonstrate different strategies for handling missing values."""
    print("\n=== Handling Missing Values ===")
    
    # Create a copy with some artificial missing values for demonstration
    df_missing = df.copy()
    
    # Randomly introduce some missing values
    np.random.seed(42)
    missing_indices = np.random.choice(df_missing.index, size=10, replace=False)
    df_missing.loc[missing_indices, 'sepal length (cm)'] = np.nan
    
    print(f"Introduced missing values: {df_missing.isnull().sum().sum()}")
    
    # Strategy 1: Remove rows with missing values
    df_dropped = df_missing.dropna()
    print(f"After dropping rows: {df_dropped.shape}")
    
    # Strategy 2: Impute with mean
    imputer_mean = SimpleImputer(strategy='mean')
    X_imputed_mean = imputer_mean.fit_transform(df_missing.select_dtypes(include=[np.number]))
    print(f"Imputed with mean - missing values: {np.isnan(X_imputed_mean).sum()}")
    
    # Strategy 3: Impute with median
    imputer_median = SimpleImputer(strategy='median')
    X_imputed_median = imputer_median.fit_transform(df_missing.select_dtypes(include=[np.number]))
    print(f"Imputed with median - missing values: {np.isnan(X_imputed_median).sum()}")
    
    return df_missing, X_imputed_mean

def feature_scaling_demo(df):
    """Demonstrate different feature scaling techniques."""
    print("\n=== Feature Scaling ===")
    
    # Select numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'species' in numerical_features:
        numerical_features.remove('species')
    
    X = df[numerical_features]
    
    print(f"Original data statistics:")
    print(X.describe())
    
    # Standard Scaling (Z-score normalization)
    scaler_standard = StandardScaler()
    X_standard = scaler_standard.fit_transform(X)
    X_standard_df = pd.DataFrame(X_standard, columns=numerical_features)
    
    print(f"\nAfter Standard Scaling:")
    print(X_standard_df.describe())
    
    # Min-Max Scaling
    scaler_minmax = MinMaxScaler()
    X_minmax = scaler_minmax.fit_transform(X)
    X_minmax_df = pd.DataFrame(X_minmax, columns=numerical_features)
    
    print(f"\nAfter Min-Max Scaling:")
    print(X_minmax_df.describe())
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data
    X.boxplot(ax=axes[0])
    axes[0].set_title('Original Data')
    axes[0].set_ylabel('Values')
    
    # Standard scaled
    X_standard_df.boxplot(ax=axes[1])
    axes[1].set_title('Standard Scaled')
    axes[1].set_ylabel('Values')
    
    # Min-max scaled
    X_minmax_df.boxplot(ax=axes[2])
    axes[2].set_title('Min-Max Scaled')
    axes[2].set_ylabel('Values')
    
    plt.tight_layout()
    plt.savefig('feature_scaling_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return X_standard_df, X_minmax_df

def categorical_encoding_demo():
    """Demonstrate categorical variable encoding techniques."""
    print("\n=== Categorical Encoding ===")
    
    # Create sample categorical data
    data = {
        'color': ['red', 'blue', 'green', 'red', 'blue', 'green'],
        'size': ['small', 'medium', 'large', 'medium', 'small', 'large'],
        'price': [10, 15, 20, 12, 8, 25]
    }
    df_cat = pd.DataFrame(data)
    print("Original categorical data:")
    print(df_cat)
    
    # Label Encoding
    label_encoder = LabelEncoder()
    df_cat['color_encoded'] = label_encoder.fit_transform(df_cat['color'])
    df_cat['size_encoded'] = label_encoder.fit_transform(df_cat['size'])
    
    print(f"\nAfter Label Encoding:")
    print(df_cat[['color', 'color_encoded', 'size', 'size_encoded']])
    
    # One-Hot Encoding
    df_onehot = pd.get_dummies(df_cat[['color', 'size', 'price']], 
                               columns=['color', 'size'], 
                               prefix=['color', 'size'])
    
    print(f"\nAfter One-Hot Encoding:")
    print(df_onehot)
    
    return df_cat, df_onehot

def main():
    """Main function to run all preprocessing examples."""
    print("🚀 Machine Learning Data Preprocessing Tutorial")
    print("=" * 50)
    
    # Load and explore data
    iris_df = load_sample_data()
    iris_df = explore_data(iris_df)
    
    # Handle missing values
    df_missing, X_imputed = handle_missing_values(iris_df)
    
    # Feature scaling
    X_standard, X_minmax = feature_scaling_demo(iris_df)
    
    # Categorical encoding
    df_cat, df_onehot = categorical_encoding_demo()
    
    print("\n✅ Data preprocessing tutorial completed!")
    print("\nKey takeaways:")
    print("1. Always explore your data first")
    print("2. Handle missing values appropriately")
    print("3. Scale numerical features for better algorithm performance")
    print("4. Encode categorical variables properly")
    print("5. Choose preprocessing techniques based on your algorithm and data")

if __name__ == "__main__":
    main()