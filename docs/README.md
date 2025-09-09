# Machine Learning with Scikit-Learn - Documentation

This directory contains additional documentation for the Machine Learning with Scikit-Learn tutorial repository.

## Quick Reference

### Installation
```bash
pip install -r requirements.txt
```

### Running Examples
```bash
# Data Preprocessing
cd examples/01_data_preprocessing
python basic_preprocessing.py

# Classification
cd examples/02_classification
python classification_algorithms.py

# Regression
cd examples/03_regression
python regression_algorithms.py

# Clustering
cd examples/04_clustering
python clustering_algorithms.py

# Model Evaluation
cd examples/05_model_evaluation
python model_evaluation.py
```

### Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/ directory and open:
# - 01_getting_started.ipynb
```

## Learning Path

### Beginner (Start Here)
1. **README.md** - Overview and setup
2. **notebooks/01_getting_started.ipynb** - Interactive introduction
3. **examples/01_data_preprocessing/** - Learn data preparation
4. **examples/02_classification/** - Your first ML models

### Intermediate
1. **examples/03_regression/** - Prediction problems
2. **examples/04_clustering/** - Unsupervised learning
3. **examples/05_model_evaluation/** - Model validation

### Advanced
1. **examples/06_advanced_topics/** - Advanced techniques
2. Experiment with different datasets
3. Try hyperparameter tuning
4. Explore ensemble methods

## Common Scikit-Learn Patterns

### Basic Workflow
```python
# 1. Import
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 2. Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Create model
model = RandomForestClassifier()

# 4. Train
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate
accuracy = accuracy_score(y_test, y_pred)
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"Average accuracy: {scores.mean():.3f}")
```

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [50, 100, 200]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Make sure you've installed requirements: `pip install -r requirements.txt`
   - Check Python version (3.7+ required)

2. **Memory Issues**
   - Reduce dataset size for large datasets
   - Use `n_jobs=1` instead of `n_jobs=-1`

3. **Performance Issues**
   - Start with smaller datasets
   - Use simpler models first
   - Consider feature selection

### Getting Help

1. Check the comments in the example scripts
2. Read the Scikit-Learn documentation
3. Open an issue in this repository
4. Check Stack Overflow for common problems

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add your examples or improvements
4. Submit a pull request

## Resources

- [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Scikit-Learn API Reference](https://scikit-learn.org/stable/modules/classes.html)
- [Machine Learning Glossary](https://developers.google.com/machine-learning/glossary)
- [Kaggle Learn](https://www.kaggle.com/learn)