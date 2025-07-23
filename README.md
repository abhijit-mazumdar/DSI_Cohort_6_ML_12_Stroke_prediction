Key Features of the Notebook:
1. Data Quality Assessment:

Identifies 201 missing BMI values (marked as "N/A")
Reveals severe class imbalance (19.52:1 ratio, only 4.87% positive cases)
Analyzes distribution of all categorical variables

2. Data Cleaning & Preprocessing:

Missing Value Imputation: Uses KNN imputation for BMI values
Feature Engineering: Creates age groups, BMI categories, glucose categories, and health risk scores
Encoding: One-hot encoding for categorical variables
Scaling: StandardScaler for numerical features

3. Class Imbalance Handling:

Implements multiple techniques: SMOTE, ADASYN, SMOTETomek, and Random Under Sampling
Compares results to help choose the best approach for modeling

4. Principal Component Analysis:

Determines that 95% of variance can be explained with fewer components
Provides both scree plots and variance analysis
Creates PCA-transformed datasets for experimentation

5. Feature Importance Analysis:

Multiple Models: Uses Random Forest and Logistic Regression
SHAP Values: Implements SHAP analysis for interpretable feature importance
Visualizations: Creates comprehensive plots showing which features most affect stroke prediction

6. Advanced Features:

Comprehensive EDA with correlation matrices and distribution plots
Engineered health risk scores combining multiple risk factors
Saves multiple versions of processed datasets for ML experimentation

Key Insights the Notebook Will Reveal:
Based on the dataset structure, the analysis will likely show that:

Age is the strongest predictor (older patients have higher stroke risk)
Heart disease and hypertension are significant risk factors
Average glucose level (diabetes indicator) is important
Work type and marriage status may show demographic patterns
The engineered health risk score may be a strong composite predictor

Ready for MLflow Integration:
The notebook saves processed datasets in multiple formats:

Original processed data
SMOTE-resampled training data
PCA-transformed data
Test set
