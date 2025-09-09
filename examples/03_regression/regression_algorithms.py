"""
Regression Examples with Scikit-Learn

This script demonstrates various regression algorithms:
- Linear Regression
- Polynomial Regression
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Decision Tree Regression
- Random Forest Regression

Author: Machine Learning Tutorial Series
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample regression dataset."""
    print("=== Creating Sample Regression Data ===")
    
    # Generate synthetic dataset
    X, y = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Visualize the data
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, alpha=0.6, label='Training data')
    plt.scatter(X_test, y_test, alpha=0.6, label='Test data')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Sample Regression Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('regression_data.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return X_train, X_test, y_train, y_test

def linear_regression_example(X_train, X_test, y_train, y_test):
    """Demonstrate Linear Regression."""
    print("\n=== Linear Regression ===")
    
    # Create and train the model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr_model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Model parameters
    print(f"Coefficient: {lr_model.coef_[0]:.4f}")
    print(f"Intercept: {lr_model.intercept_:.4f}")
    
    return lr_model, y_pred, r2

def polynomial_regression_example(X_train, X_test, y_train, y_test):
    """Demonstrate Polynomial Regression."""
    print("\n=== Polynomial Regression ===")
    
    # Transform features to polynomial features
    poly_features = PolynomialFeatures(degree=3)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    # Create and train the model
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    
    # Make predictions
    y_pred = poly_model.predict(X_test_poly)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Number of polynomial features: {X_train_poly.shape[1]}")
    
    return poly_model, y_pred, r2, poly_features

def ridge_regression_example(X_train, X_test, y_train, y_test):
    """Demonstrate Ridge Regression with L2 regularization."""
    print("\n=== Ridge Regression (L2 Regularization) ===")
    
    # Scale features for regularization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = ridge_model.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Alpha (regularization strength): {ridge_model.alpha}")
    
    return ridge_model, y_pred, r2

def lasso_regression_example(X_train, X_test, y_train, y_test):
    """Demonstrate Lasso Regression with L1 regularization."""
    print("\n=== Lasso Regression (L1 Regularization) ===")
    
    # Scale features for regularization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    lasso_model = Lasso(alpha=0.1, random_state=42)
    lasso_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = lasso_model.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Alpha (regularization strength): {lasso_model.alpha}")
    
    return lasso_model, y_pred, r2

def decision_tree_regression_example(X_train, X_test, y_train, y_test):
    """Demonstrate Decision Tree Regression."""
    print("\n=== Decision Tree Regression ===")
    
    # Create and train the model
    dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = dt_model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Max depth: {dt_model.max_depth}")
    
    return dt_model, y_pred, r2

def random_forest_regression_example(X_train, X_test, y_train, y_test):
    """Demonstrate Random Forest Regression."""
    print("\n=== Random Forest Regression ===")
    
    # Create and train the model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Number of trees: {rf_model.n_estimators}")
    
    return rf_model, y_pred, r2

def compare_models(X_train, X_test, y_train, y_test):
    """Compare all regression models."""
    print("\n=== Model Comparison ===")
    
    # Dictionary to store models and their results
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=0.1, random_state=42),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    predictions = {}
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for name, model in models.items():
        # Some models need scaled features
        if name in ['Ridge Regression', 'Lasso Regression']:
            X_train_processed = X_train_scaled
            X_test_processed = X_test_scaled
        else:
            X_train_processed = X_train
            X_test_processed = X_test
        
        # Train and evaluate
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results[name] = {'R² Score': r2, 'RMSE': rmse}
        predictions[name] = y_pred
        
        print(f"{name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # R² Score comparison
    models_list = list(results.keys())
    r2_scores = [results[model]['R² Score'] for model in models_list]
    
    bars1 = ax1.bar(models_list, r2_scores, color='skyblue')
    ax1.set_title('R² Score Comparison')
    ax1.set_ylabel('R² Score')
    ax1.set_ylim(0, 1)
    
    # Add values on top of bars
    for bar, score in zip(bars1, r2_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    ax1.tick_params(axis='x', rotation=45)
    
    # RMSE comparison
    rmse_scores = [results[model]['RMSE'] for model in models_list]
    
    bars2 = ax2.bar(models_list, rmse_scores, color='lightcoral')
    ax2.set_title('RMSE Comparison')
    ax2.set_ylabel('RMSE')
    
    # Add values on top of bars
    for bar, score in zip(bars2, rmse_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{score:.1f}', ha='center', va='bottom')
    
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('regression_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results, predictions

def plot_predictions_comparison(X_test, y_test, predictions):
    """Plot predictions vs actual values for different models."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, (model_name, y_pred) in enumerate(predictions.items()):
        axes[i].scatter(y_test, y_pred, alpha=0.6)
        axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[i].set_xlabel('Actual Values')
        axes[i].set_ylabel('Predicted Values')
        axes[i].set_title(f'{model_name}')
        axes[i].grid(True, alpha=0.3)
        
        # Calculate and display R²
        r2 = r2_score(y_test, y_pred)
        axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes, 
                    bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    # Hide the last subplot if we have an odd number of models
    if len(predictions) < 6:
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run all regression examples."""
    print("📈 Machine Learning Regression Tutorial")
    print("=" * 50)
    
    # Create and prepare data
    X_train, X_test, y_train, y_test = create_sample_data()
    
    # Run individual regression examples
    lr_model, lr_pred, lr_r2 = linear_regression_example(X_train, X_test, y_train, y_test)
    poly_model, poly_pred, poly_r2, poly_features = polynomial_regression_example(X_train, X_test, y_train, y_test)
    ridge_model, ridge_pred, ridge_r2 = ridge_regression_example(X_train, X_test, y_train, y_test)
    lasso_model, lasso_pred, lasso_r2 = lasso_regression_example(X_train, X_test, y_train, y_test)
    dt_model, dt_pred, dt_r2 = decision_tree_regression_example(X_train, X_test, y_train, y_test)
    rf_model, rf_pred, rf_r2 = random_forest_regression_example(X_train, X_test, y_train, y_test)
    
    # Compare all models
    results, predictions = compare_models(X_train, X_test, y_train, y_test)
    
    # Plot predictions comparison
    plot_predictions_comparison(X_test, y_test, predictions)
    
    # Find best model
    best_model = max(results, key=lambda x: results[x]['R² Score'])
    print(f"\n🏆 Best Model: {best_model}")
    print(f"R² Score: {results[best_model]['R² Score']:.4f}")
    print(f"RMSE: {results[best_model]['RMSE']:.4f}")
    
    print("\n✅ Regression tutorial completed!")
    print("\nKey takeaways:")
    print("1. Linear regression works well for linear relationships")
    print("2. Polynomial regression can capture non-linear patterns")
    print("3. Ridge/Lasso regression help prevent overfitting with regularization")
    print("4. Tree-based models can capture complex non-linear relationships")
    print("5. Ensemble methods (Random Forest) often provide better performance")
    print("6. Always evaluate models using appropriate metrics (R², RMSE, MAE)")

if __name__ == "__main__":
    main()