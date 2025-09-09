"""
Model Evaluation and Validation with Scikit-Learn

This script demonstrates various model evaluation techniques:
- Cross-validation strategies
- Performance metrics for classification and regression
- Learning curves
- Validation curves
- Grid search for hyperparameter tuning

Author: Machine Learning Tutorial Series
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_boston, make_classification
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                     cross_validate, KFold, StratifiedKFold,
                                     learning_curve, validation_curve,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix,
                            mean_squared_error, r2_score, mean_absolute_error)
import warnings
warnings.filterwarnings('ignore')

def load_classification_data():
    """Load and prepare classification dataset."""
    print("=== Loading Classification Data ===")
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, iris.target_names

def cross_validation_example(X_train, y_train):
    """Demonstrate different cross-validation strategies."""
    print("\n=== Cross-Validation Strategies ===")
    
    # Create a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 1. Simple cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Individual fold scores: {cv_scores}")
    
    # 2. Stratified K-Fold (for classification)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stratified_scores = cross_val_score(model, X_train, y_train, cv=skf)
    print(f"Stratified 5-Fold CV Accuracy: {stratified_scores.mean():.4f} (+/- {stratified_scores.std() * 2:.4f})")
    
    # 3. Cross-validate with multiple metrics
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
    
    print("\nDetailed Cross-Validation Results:")
    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        print(f"{metric.capitalize()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return cv_scores, cv_results

def classification_metrics_example(X_train, X_test, y_train, y_test, target_names):
    """Demonstrate classification performance metrics."""
    print("\n=== Classification Performance Metrics ===")
    
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-Score (macro): {f1:.4f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, precision, recall, f1

def learning_curves_example(X_train, y_train):
    """Demonstrate learning curves for model evaluation."""
    print("\n=== Learning Curves ===")
    
    # Create models with different complexities
    models = {
        'Simple Model': RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42),
        'Complex Model': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    plt.figure(figsize=(15, 5))
    
    for i, (name, model) in enumerate(models.items()):
        # Calculate learning curve
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        plt.subplot(1, 2, i+1)
        plt.plot(train_sizes, train_mean, 'o-', label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std, 
                        train_mean + train_std, alpha=0.1)
        
        plt.plot(train_sizes, val_mean, 'o-', label='Validation score')
        plt.fill_between(train_sizes, val_mean - val_std, 
                        val_mean + val_std, alpha=0.1)
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title(f'Learning Curve - {name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def validation_curves_example(X_train, y_train):
    """Demonstrate validation curves for hyperparameter tuning."""
    print("\n=== Validation Curves ===")
    
    # Example: Random Forest n_estimators
    param_name = 'n_estimators'
    param_range = [10, 50, 100, 200, 300, 500]
    
    train_scores, val_scores = validation_curve(
        RandomForestClassifier(random_state=42), X_train, y_train,
        param_name=param_name, param_range=param_range, cv=5
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot validation curve
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, 'o-', label='Training score')
    plt.fill_between(param_range, train_mean - train_std, 
                    train_mean + train_std, alpha=0.1)
    
    plt.plot(param_range, val_mean, 'o-', label='Validation score')
    plt.fill_between(param_range, val_mean - val_std, 
                    val_mean + val_std, alpha=0.1)
    
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy Score')
    plt.title('Validation Curve - Random Forest')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('validation_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find optimal parameter
    optimal_param = param_range[np.argmax(val_mean)]
    print(f"Optimal {param_name}: {optimal_param}")
    
    return optimal_param

def grid_search_example(X_train, y_train):
    """Demonstrate Grid Search for hyperparameter tuning."""
    print("\n=== Grid Search Cross-Validation ===")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # Create model
    rf = RandomForestClassifier(random_state=42)
    
    # Grid search
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    print("Performing Grid Search...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get results as DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)
    print(f"\nTotal combinations tested: {len(results_df)}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Heatmap of mean test scores
    pivot_table = results_df.pivot_table(
        values='mean_test_score', 
        index='param_max_depth', 
        columns='param_n_estimators'
    )
    
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Grid Search Results - Mean Test Score')
    plt.tight_layout()
    plt.savefig('grid_search_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return grid_search.best_estimator_, grid_search.best_params_

def randomized_search_example(X_train, y_train):
    """Demonstrate Randomized Search for hyperparameter tuning."""
    print("\n=== Randomized Search Cross-Validation ===")
    
    # Define parameter distributions
    param_distributions = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    
    # Create model
    rf = RandomForestClassifier(random_state=42)
    
    # Randomized search
    random_search = RandomizedSearchCV(
        rf, param_distributions, n_iter=50, cv=5, 
        scoring='accuracy', n_jobs=-1, random_state=42, verbose=1
    )
    
    print("Performing Randomized Search...")
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_

def model_comparison_example(X_train, X_test, y_train, y_test):
    """Compare different models with proper evaluation."""
    print("\n=== Model Comparison with Cross-Validation ===")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Train on full training set and test
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        results[name] = {
            'CV Mean': cv_scores.mean(),
            'CV Std': cv_scores.std(),
            'Test Score': test_score
        }
        
        print(f"{name}:")
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"  Test Score: {test_score:.4f}")
    
    # Plot comparison
    models_list = list(results.keys())
    cv_means = [results[model]['CV Mean'] for model in models_list]
    cv_stds = [results[model]['CV Std'] for model in models_list]
    test_scores = [results[model]['Test Score'] for model in models_list]
    
    x = np.arange(len(models_list))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    
    # CV scores with error bars
    plt.bar(x - width/2, cv_means, width, yerr=cv_stds, 
            label='Cross-Validation', alpha=0.7, capsize=5)
    
    # Test scores
    plt.bar(x + width/2, test_scores, width, 
            label='Test Score', alpha=0.7)
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison: Cross-Validation vs Test Scores')
    plt.xticks(x, models_list)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (cv_mean, test_score) in enumerate(zip(cv_means, test_scores)):
        plt.text(i - width/2, cv_mean + 0.01, f'{cv_mean:.3f}', 
                ha='center', va='bottom')
        plt.text(i + width/2, test_score + 0.01, f'{test_score:.3f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def regression_metrics_example():
    """Demonstrate regression evaluation metrics."""
    print("\n=== Regression Evaluation Metrics ===")
    
    # Create synthetic regression data
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual Values')
    plt.grid(True, alpha=0.3)
    
    # Residuals plot
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run all evaluation examples."""
    print("📊 Machine Learning Model Evaluation Tutorial")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test, target_names = load_classification_data()
    
    # Cross-validation examples
    cv_scores, cv_results = cross_validation_example(X_train, y_train)
    
    # Classification metrics
    accuracy, precision, recall, f1 = classification_metrics_example(
        X_train, X_test, y_train, y_test, target_names
    )
    
    # Learning curves
    learning_curves_example(X_train, y_train)
    
    # Validation curves
    optimal_param = validation_curves_example(X_train, y_train)
    
    # Grid search
    best_model_grid, best_params_grid = grid_search_example(X_train, y_train)
    
    # Randomized search
    best_model_random, best_params_random = randomized_search_example(X_train, y_train)
    
    # Model comparison
    comparison_results = model_comparison_example(X_train, X_test, y_train, y_test)
    
    # Regression metrics
    regression_metrics_example()
    
    print("\n✅ Model evaluation tutorial completed!")
    print("\nKey takeaways:")
    print("1. Always use cross-validation to get reliable performance estimates")
    print("2. Use appropriate metrics for your problem (accuracy, precision, recall, F1)")
    print("3. Learning curves help diagnose overfitting and underfitting")
    print("4. Validation curves help find optimal hyperparameters")
    print("5. Grid search explores all parameter combinations systematically")
    print("6. Randomized search is more efficient for large parameter spaces")
    print("7. Compare multiple models using the same evaluation protocol")
    print("8. For regression, use MSE, RMSE, MAE, and R² as evaluation metrics")

if __name__ == "__main__":
    main()