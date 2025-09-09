"""
Classification Examples with Scikit-Learn

This script demonstrates various classification algorithms:
- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machine (SVM)
- k-Nearest Neighbors (k-NN)

Author: Machine Learning Tutorial Series
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the Iris dataset for classification."""
    print("=== Loading and Preparing Data ===")
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(target_names)}")
    print(f"Classes: {target_names}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, feature_names, target_names

def logistic_regression_example(X_train, X_test, y_train, y_test):
    """Demonstrate Logistic Regression classification."""
    print("\n=== Logistic Regression ===")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = lr_model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return lr_model, y_pred, accuracy

def decision_tree_example(X_train, X_test, y_train, y_test):
    """Demonstrate Decision Tree classification."""
    print("\n=== Decision Tree ===")
    
    # Create and train the model
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=3)
    dt_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = dt_model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Feature importance
    print("\nFeature importance:")
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for i, importance in enumerate(dt_model.feature_importances_):
        print(f"{feature_names[i]}: {importance:.4f}")
    
    return dt_model, y_pred, accuracy

def random_forest_example(X_train, X_test, y_train, y_test):
    """Demonstrate Random Forest classification."""
    print("\n=== Random Forest ===")
    
    # Create and train the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Feature importance
    print("\nFeature importance:")
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for i, importance in enumerate(rf_model.feature_importances_):
        print(f"{feature_names[i]}: {importance:.4f}")
    
    return rf_model, y_pred, accuracy

def svm_example(X_train, X_test, y_train, y_test):
    """Demonstrate Support Vector Machine classification."""
    print("\n=== Support Vector Machine ===")
    
    # Scale the features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = svm_model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    return svm_model, y_pred, accuracy

def knn_example(X_train, X_test, y_train, y_test):
    """Demonstrate k-Nearest Neighbors classification."""
    print("\n=== k-Nearest Neighbors ===")
    
    # Scale the features (important for k-NN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = knn_model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    return knn_model, y_pred, accuracy

def compare_models(X_train, X_test, y_train, y_test, target_names):
    """Compare all classification models."""
    print("\n=== Model Comparison ===")
    
    # Dictionary to store models and their results
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=3),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'k-NN': KNeighborsClassifier(n_neighbors=3)
    }
    
    results = {}
    
    for name, model in models.items():
        # Some models need scaled features
        if name in ['Logistic Regression', 'SVM', 'k-NN']:
            scaler = StandardScaler()
            X_train_processed = scaler.fit_transform(X_train)
            X_test_processed = scaler.transform(X_test)
        else:
            X_train_processed = X_train
            X_test_processed = X_test
        
        # Train and evaluate
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = accuracy
        print(f"{name}: {accuracy:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    models_list = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(models_list, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightpink'])
    plt.title('Classification Model Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add accuracy values on top of bars
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{accuracy:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def plot_confusion_matrix(y_test, y_pred, target_names):
    """Plot confusion matrix for model evaluation."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run all classification examples."""
    print("🎯 Machine Learning Classification Tutorial")
    print("=" * 50)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names, target_names = load_and_prepare_data()
    
    # Run individual classification examples
    lr_model, lr_pred, lr_acc = logistic_regression_example(X_train, X_test, y_train, y_test)
    dt_model, dt_pred, dt_acc = decision_tree_example(X_train, X_test, y_train, y_test)
    rf_model, rf_pred, rf_acc = random_forest_example(X_train, X_test, y_train, y_test)
    svm_model, svm_pred, svm_acc = svm_example(X_train, X_test, y_train, y_test)
    knn_model, knn_pred, knn_acc = knn_example(X_train, X_test, y_train, y_test)
    
    # Compare all models
    results = compare_models(X_train, X_test, y_train, y_test, target_names)
    
    # Show detailed classification report for best model
    best_model = max(results, key=results.get)
    print(f"\n📊 Detailed Report for Best Model: {best_model}")
    
    # Use Random Forest predictions as an example
    print("\nClassification Report:")
    print(classification_report(y_test, rf_pred, target_names=target_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, rf_pred, target_names)
    
    print("\n✅ Classification tutorial completed!")
    print("\nKey takeaways:")
    print("1. Different algorithms work better for different types of data")
    print("2. Feature scaling is important for some algorithms (SVM, k-NN, Logistic Regression)")
    print("3. Tree-based models (Decision Tree, Random Forest) don't require feature scaling")
    print("4. Ensemble methods (Random Forest) often perform better than single models")
    print("5. Always evaluate models using appropriate metrics")

if __name__ == "__main__":
    main()