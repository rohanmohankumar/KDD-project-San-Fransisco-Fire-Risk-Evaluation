# San Francisco Fire Severity Prediction - KDD Project

A comprehensive machine learning project for predicting fire incident severity in San Francisco using historical incident data. This project implements multi-class classification with extensive exploratory data analysis (EDA) and model comparison.

## Project Overview

This project analyzes 538,285 fire incidents to predict severity levels (Low, Medium, High) using various machine learning algorithms. The analysis includes feature engineering, temporal analysis, and comparison of multiple classification models.

### Target Variable: Fire Severity
- **Low (Class 0)**: 1-2 units, ≤30 min response, 1 alarm (43.78%)
- **Medium (Class 1)**: 3-5 units, 30-90 min response, 1-2 alarms (47.68%)
- **High (Class 2)**: ≥6 units, ≥90 min response, ≥3 alarms (8.54%)

## Key Features

### Data Processing
- **Dataset Size**: 538,285 incidents × 80 features
- **Final Feature Matrix**: 538,164 incidents × 29 engineered features
- **Train-Test Split**: 75%-25% stratified split

### Feature Engineering
1. **Temporal Features**:
   - Hour, DayOfWeek, Month, Season
   - IsWeekend, IsPeakHour flags

2. **Categorical Features**:
   - Neighborhood analysis
   - Battalion classification
   - Situation categorization (Fire, Alarm, Accident, Medical)
   - Property type categorization (Residential, Street, Commercial)

3. **Aggregate Features**:
   - Neighborhood fire frequency
   - Neighborhood fire count

4. **Response Metrics**:
   - Suppression units/personnel
   - EMS units/personnel
   - Response duration

## Machine Learning Models

### Models Implemented
| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| **Gradient Boosting** | **88.24%** | **0.8813** | 156.33s |
| Decision Tree | 84.73% | 0.8519 | 1.18s |
| Random Forest | 84.52% | 0.8494 | 3.87s |
| Logistic Regression | 81.37% | 0.8161 | 1.65s |

### Performance Metrics (Best Model: Gradient Boosting)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Low | 0.85 | 0.94 | 0.89 |
| Medium | 0.91 | 0.86 | 0.88 |
| High | 0.96 | 0.70 | 0.81 |

### Confusion Matrix Analysis
The Gradient Boosting model demonstrates:
- Excellent performance on Low severity incidents (94% recall)
- Strong Medium severity prediction (86% recall)
- Good High severity detection (70% recall) with 96% precision

## Exploratory Data Analysis

### Key Findings
1. **Temporal Patterns**:
   - Peak incident hours: 6-9 AM and 5-8 PM
   - Weekend variations in incident types
   - Seasonal trends throughout the year

2. **Severity Correlations**:
   - Response duration strongly correlated with severity
   - Number of suppression units indicates severity level
   - Alarm count is a significant severity indicator

3. **Geographic Distribution**:
   - Certain neighborhoods show higher incident frequencies
   - Battalion assignments influence response patterns

## 🛠️ Installation & Requirements

```bash
# Core Dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Specific versions (recommended)
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Usage

### Running the Analysis

```python
# Load and preprocess data
import pandas as pd
df = pd.read_csv('fire-incidents.csv')

# The notebook handles:
# 1. Data loading and preprocessing
# 2. Target variable creation
# 3. Feature engineering
# 4. Model training
# 5. Evaluation and visualization

# Simply execute:
python KDD_project.ipynb  # or run in Jupyter
```

### Quick Start Example

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# After feature engineering (see notebook for details)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train best performing model
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)

# Evaluate
accuracy = gb_model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

## Project Structure

```
.
├── KDD_project.ipynb              # Main analysis notebook
├── fire-incidents.csv             # Dataset (not included)
├── fire_severity_model_comparison.csv  # Model results
├── feature_importance.csv          # Feature rankings
├── README.md                       # This file
└── visualizations/                 # Generated plots
    ├── eda_analysis.png
    ├── confusion_matrices.png
    ├── model_comparison.png
    └── feature_importance.png
```

## Visualizations

The project generates comprehensive visualizations:

1. **EDA Visualizations**:
   - Class distribution
   - Temporal patterns (hour, day, month)
   - Severity correlations
   - Feature correlation heatmap

2. **Model Performance**:
   - Confusion matrices (all models)
   - Accuracy comparison bars
   - F1-score comparison
   - Feature importance rankings

## Feature Importance (Top 15 - Random Forest)

The most influential features for prediction include:
- Response duration
- Number of suppression units
- Number of alarms
- Suppression personnel count
- Temporal features (hour, day)
- Neighborhood characteristics

## Key Insights

### Model Selection
**Gradient Boosting** emerges as the best model with:
- Highest overall accuracy (88.24%)
- Balanced precision-recall trade-off
- Excellent performance on majority classes
- Strong High severity detection (96% precision)

### Practical Applications
1. **Resource Allocation**: Predict required units based on incident characteristics
2. **Response Planning**: Anticipate severity for optimal dispatch
3. **Risk Assessment**: Identify high-risk neighborhoods and times
4. **Training**: Focus areas where prediction confidence is lower

### Trade-offs
- **Gradient Boosting**: Best accuracy but slower training (156s)
- **Decision Tree**: Fast training (1.18s) with competitive accuracy (84.73%)
- **Random Forest**: Good balance of speed and performance

## Methodology

### Data Pipeline
1. **Data Loading**: Parse datetime fields, handle missing values
2. **Target Creation**: Engineer severity labels from incident metrics
3. **Feature Engineering**: Create temporal, categorical, and aggregate features
4. **Preprocessing**: Standard scaling, one-hot encoding
5. **Model Training**: Train multiple classifiers with class balancing
6. **Evaluation**: Comprehensive metrics, confusion matrices, cross-validation

### Evaluation Metrics
- Accuracy
- Precision (Macro & Weighted)
- Recall (Macro & Weighted)
- F1-Score (Macro & Weighted)
- Confusion Matrices
- Training Time

## Academic Context

This project was developed as part of a Knowledge Discovery and Data Mining (KDDM/CS 513-A) course, demonstrating:
- End-to-end ML pipeline development
- Feature engineering for real-world data
- Model selection and evaluation
- Data visualization and interpretation
- Domain-specific problem solving

## Future Improvements

1. **Model Enhancements**:
   - Hyperparameter tuning (GridSearchCV)
   - Ensemble methods (stacking, voting)
   - Neural network architectures
   - Deep learning approaches

2. **Feature Engineering**:
   - Geographic clustering
   - Weather data integration
   - Historical incident patterns
   - Time-series analysis

3. **Deployment**:
   - Real-time prediction API
   - Dashboard for monitoring
   - Mobile application integration
   - Alert system for high-severity predictions

## Data Source

Dataset: San Francisco Fire Department Incident Data
- Source: San Francisco Open Data Portal
- Time Period: Multi-year historical records
- Size: 538,285 incidents
- Features: 80 original attributes

## Contributing

This is an academic project. For similar analyses or improvements:
1. Fork the repository
2. Create feature branches
3. Submit pull requests with detailed descriptions
4. Ensure code follows PEP 8 style guidelines

## 📧 Contact

**Rohan Mohankumar**  
Master's in Computer Science  
Specialization: Machine Learning, Data Mining, NLP

## Acknowledgments

- San Francisco Fire Department for the dataset
- Course: CS 513-A - Knowledge Discovery and Data Mining
- Libraries: scikit-learn, pandas, NumPy, matplotlib, seaborn

## License

Academic project - Please contact author for usage permissions.

---

## Quick Links

- [Model Comparison Results](fire_severity_model_comparison.csv)
- [Feature Importance Analysis](feature_importance.csv)
- [Full Jupyter Notebook](KDD_project.ipynb)

---

**Note**: This README provides a comprehensive overview of the fire severity prediction project. For detailed implementation, refer to the Jupyter notebook.
