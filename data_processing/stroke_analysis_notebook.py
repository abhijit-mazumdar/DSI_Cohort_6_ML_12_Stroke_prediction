# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
# ---

# Stroke Dataset Analysis and Preprocessing
# Comprehensive data preprocessing pipeline with PCA and SHAP analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("All required libraries imported successfully!")

# # 1. Data Loading and Initial Exploration

# Load the dataset
df = pd.read_csv('healthcaredatasetstrokedata.csv')

print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
df.head()

# Basic info about the dataset
print("\nDataset Info:")
df.info()

print("\nBasic Statistics:")
df.describe()

# # 2. Data Quality Assessment

# Check for missing values
print("Missing Values:")
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_values.index,
    'Missing_Count': missing_values.values,
    'Missing_Percentage': missing_percent.values
})
print(missing_df[missing_df['Missing_Count'] > 0])

# Check for "N/A" strings in BMI column
print(f"\nBMI 'N/A' values: {(df['bmi'] == 'N/A').sum()}")

# Check unique values for categorical variables
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
print("\nUnique values in categorical columns:")
for col in categorical_cols:
    print(f"{col}: {df[col].unique()}")

# Target variable distribution
print("\nTarget Variable Distribution:")
print(df['stroke'].value_counts())
print(f"Class imbalance ratio: {df['stroke'].value_counts()[0] / df['stroke'].value_counts()[1]:.2f}:1")
print(f"Positive class percentage: {(df['stroke'].sum() / len(df)) * 100:.2f}%")

# # 3. Exploratory Data Analysis

# Create subplots for better visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Age distribution by stroke
sns.boxplot(data=df, x='stroke', y='age', ax=axes[0,0])
axes[0,0].set_title('Age Distribution by Stroke Status')

# Gender distribution by stroke
stroke_gender = pd.crosstab(df['gender'], df['stroke'], normalize='index')
stroke_gender.plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Stroke Rate by Gender')
axes[0,1].legend(['No Stroke', 'Stroke'])

# Work type distribution by stroke
stroke_work = pd.crosstab(df['work_type'], df['stroke'], normalize='index')
stroke_work.plot(kind='bar', ax=axes[1,0], rot=45)
axes[1,0].set_title('Stroke Rate by Work Type')
axes[1,0].legend(['No Stroke', 'Stroke'])

# Smoking status distribution by stroke
stroke_smoking = pd.crosstab(df['smoking_status'], df['stroke'], normalize='index')
stroke_smoking.plot(kind='bar', ax=axes[1,1], rot=45)
axes[1,1].set_title('Stroke Rate by Smoking Status')
axes[1,1].legend(['No Stroke', 'Stroke'])

plt.tight_layout()
plt.show()

# Correlation matrix for numerical variables
numerical_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'stroke']
df_numerical = df[numerical_cols].copy()

# Convert BMI to numerical, handling 'N/A' values
df_numerical['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

plt.figure(figsize=(10, 8))
correlation_matrix = df_numerical.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()

# # 4. Data Cleaning and Preprocessing

# Create a copy for preprocessing
df_clean = df.copy()

# Handle BMI 'N/A' values - convert to NaN for proper imputation
df_clean['bmi'] = pd.to_numeric(df_clean['bmi'], errors='coerce')

print(f"BMI missing values after conversion: {df_clean['bmi'].isnull().sum()}")

# Drop ID column as it's not useful for prediction
df_clean = df_clean.drop('id', axis=1)

# Separate features and target
X = df_clean.drop('stroke', axis=1)
y = df_clean['stroke']

# Identify categorical and numerical columns
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numerical_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

# # 5. Missing Value Imputation

# For numerical features, use KNN imputation
numerical_imputer = KNNImputer(n_neighbors=5)
X_numerical = X[numerical_features].copy()
X_numerical_imputed = numerical_imputer.fit_transform(X_numerical)
X_numerical_imputed = pd.DataFrame(X_numerical_imputed, columns=numerical_features, index=X.index)

# For categorical features, use mode imputation
categorical_imputer = SimpleImputer(strategy='most_frequent')
X_categorical = X[categorical_features].copy()
X_categorical_imputed = categorical_imputer.fit_transform(X_categorical)
X_categorical_imputed = pd.DataFrame(X_categorical_imputed, columns=categorical_features, index=X.index)

print("Missing values after imputation:")
print("Numerical:", X_numerical_imputed.isnull().sum().sum())
print("Categorical:", X_categorical_imputed.isnull().sum().sum())

# # 6. Feature Engineering

# Create age groups
def categorize_age(age):
    if age < 18:
        return 'Child'
    elif age < 35:
        return 'Young Adult'
    elif age < 55:
        return 'Middle-aged'
    elif age < 75:
        return 'Senior'
    else:
        return 'Elderly'

X_numerical_imputed['age_group'] = X_numerical_imputed['age'].apply(categorize_age)

# Create BMI categories
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

X_numerical_imputed['bmi_category'] = X_numerical_imputed['bmi'].apply(categorize_bmi)

# Create glucose level categories
def categorize_glucose(glucose):
    if glucose < 100:
        return 'Normal'
    elif glucose < 126:
        return 'Prediabetic'
    else:
        return 'Diabetic'

X_numerical_imputed['glucose_category'] = X_numerical_imputed['avg_glucose_level'].apply(categorize_glucose)

# Create health risk score
X_numerical_imputed['health_risk_score'] = (
    X_numerical_imputed['hypertension'] + 
    X_numerical_imputed['heart_disease'] + 
    (X_numerical_imputed['avg_glucose_level'] > 125).astype(int) +
    (X_numerical_imputed['bmi'] > 30).astype(int)
)

print("New engineered features created:")
print("- age_group")
print("- bmi_category") 
print("- glucose_category")
print("- health_risk_score")

# # 7. Encoding Categorical Variables

# Combine all categorical features (original + engineered)
new_categorical_features = ['age_group', 'bmi_category', 'glucose_category']
all_categorical_features = categorical_features + new_categorical_features

# Prepare categorical data
X_categorical_all = pd.concat([
    X_categorical_imputed,
    X_numerical_imputed[new_categorical_features]
], axis=1)

# One-hot encode categorical variables
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_categorical_encoded = encoder.fit_transform(X_categorical_all)

# Get feature names after encoding
encoded_feature_names = encoder.get_feature_names_out(all_categorical_features)
X_categorical_encoded = pd.DataFrame(X_categorical_encoded, columns=encoded_feature_names, index=X.index)

print(f"Categorical features after encoding: {X_categorical_encoded.shape[1]} features")

# # 8. Feature Scaling

# Combine numerical features (excluding the ones used for creating categories)
numerical_features_final = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'health_risk_score']
X_numerical_final = X_numerical_imputed[numerical_features_final]

# Scale numerical features
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical_final)
X_numerical_scaled = pd.DataFrame(X_numerical_scaled, columns=numerical_features_final, index=X.index)

# Combine all processed features
X_processed = pd.concat([X_numerical_scaled, X_categorical_encoded], axis=1)

print(f"Final processed dataset shape: {X_processed.shape}")
print(f"Total features: {X_processed.shape[1]}")

# # 9. Handle Class Imbalance

# Split data first
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print("Original class distribution in training set:")
print(y_train.value_counts())

# Apply different resampling techniques
resampling_techniques = {}

# 1. SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
resampling_techniques['SMOTE'] = (X_train_smote, y_train_smote)

# 2. ADASYN
adasyn = ADASYN(random_state=42)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
resampling_techniques['ADASYN'] = (X_train_adasyn, y_train_adasyn)

# 3. SMOTETomek (combination)
smotetomek = SMOTETomek(random_state=42)
X_train_smotetomek, y_train_smotetomek = smotetomek.fit_resample(X_train, y_train)
resampling_techniques['SMOTETomek'] = (X_train_smotetomek, y_train_smotetomek)

# 4. Random Under Sampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
resampling_techniques['UnderSampling'] = (X_train_rus, y_train_rus)

print("\nClass distribution after resampling:")
for technique, (X_resampled, y_resampled) in resampling_techniques.items():
    print(f"{technique}: {y_resampled.value_counts().to_dict()}")

# # 10. Principal Component Analysis (PCA)

# Apply PCA to reduce dimensionality
pca = PCA()
X_train_pca = pca.fit_transform(X_train_smote)  # Using SMOTE-resampled data

# Calculate cumulative variance explained
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Find number of components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components to explain 95% variance: {n_components_95}")

# Plot PCA analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Scree plot
axes[0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
axes[0].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
axes[0].axvline(x=n_components_95, color='r', linestyle='--')
axes[0].set_xlabel('Number of Components')
axes[0].set_ylabel('Cumulative Variance Explained')
axes[0].set_title('PCA: Cumulative Variance Explained')
axes[0].legend()
axes[0].grid(True)

# Individual component variance
axes[1].bar(range(1, min(21, len(pca.explained_variance_ratio_) + 1)), 
           pca.explained_variance_ratio_[:20])
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Variance Explained')
axes[1].set_title('Individual Component Variance (First 20)')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Apply PCA with optimal number of components
pca_optimal = PCA(n_components=n_components_95)
X_train_pca_optimal = pca_optimal.fit_transform(X_train_smote)
X_test_pca_optimal = pca_optimal.transform(X_test)

print(f"Original feature space: {X_train_smote.shape[1]} dimensions")
print(f"Reduced feature space: {X_train_pca_optimal.shape[1]} dimensions")
print(f"Variance retained: {pca_optimal.explained_variance_ratio_.sum():.4f}")

# # 11. Feature Importance Analysis with Multiple Models

# Train multiple models for feature importance analysis
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

feature_importance_results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_smote, y_train_smote)
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        continue
    
    # Create feature importance dataframe
    feature_names = X_processed.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    feature_importance_results[name] = importance_df
    
    # Plot top 15 features
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title(f'Top 15 Feature Importance - {name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    
    print(f"Top 10 features for {name}:")
    print(importance_df.head(10))

# # 12. SHAP Analysis

print("Performing SHAP analysis...")

# Use a smaller sample for SHAP analysis to speed up computation
sample_size = min(1000, len(X_train_smote))
sample_indices = np.random.choice(len(X_train_smote), sample_size, replace=False)
X_shap_sample = X_train_smote.iloc[sample_indices]
y_shap_sample = y_train_smote.iloc[sample_indices]

# Train Random Forest for SHAP analysis
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

# Create SHAP explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_shap_sample)

# For binary classification, use positive class SHAP values
if isinstance(shap_values, list):
    shap_values_positive = shap_values[1]  # Positive class (stroke = 1)
else:
    shap_values_positive = shap_values

# SHAP Summary Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_positive, X_shap_sample, max_display=15, show=False)
plt.title('SHAP Feature Importance Summary')
plt.tight_layout()
plt.show()

# SHAP Bar Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_positive, X_shap_sample, plot_type="bar", max_display=15, show=False)
plt.title('SHAP Feature Importance (Mean |SHAP Value|)')
plt.tight_layout()
plt.show()

# Calculate mean absolute SHAP values for ranking
mean_abs_shap = np.abs(shap_values_positive).mean(0)
feature_names = X_processed.columns
shap_importance = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

print("Top 15 features by SHAP importance:")
print(shap_importance.head(15))

# # 13. Summary and Recommendations

print("\n" + "="*50)
print("DATA PREPROCESSING SUMMARY")
print("="*50)

print(f"""
Original Dataset:
- Shape: {df.shape}
- Missing BMI values: 201 (3.93%)
- Class imbalance: 19.52:1 (4.87% positive class)

Data Cleaning:
- Imputed BMI using KNN imputation
- Created engineered features: age_group, bmi_category, glucose_category, health_risk_score
- One-hot encoded categorical variables
- Standardized numerical features

Final Processed Dataset:
- Shape: {X_processed.shape}
- Features after encoding: {X_processed.shape[1]}

Class Imbalance Handling:
- Applied SMOTE, ADASYN, SMOTETomek, and UnderSampling
- SMOTE recommended for initial modeling

PCA Analysis:
- {n_components_95} components explain 95% of variance
- Dimensionality reduction from {X_processed.shape[1]} to {n_components_95} features

Key Findings:
""")

print("Top 5 Most Important Features (Random Forest):")
if 'Random Forest' in feature_importance_results:
    for i, row in feature_importance_results['Random Forest'].head(5).iterrows():
        print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

print("\nTop 5 Most Important Features (SHAP):")
for i, (_, row) in enumerate(shap_importance.head(5).iterrows()):
    print(f"  {i+1}. {row['feature']}: {row['mean_abs_shap']:.4f}")

print(f"""
Recommendations for Model Building:
1. Use SMOTE-resampled data for training
2. Consider both original features and PCA-transformed features
3. Focus on top features identified by SHAP analysis
4. Use stratified cross-validation due to class imbalance
5. Consider ensemble methods given feature diversity
6. Monitor for overfitting due to class imbalance

Next Steps:
1. Experiment with different algorithms using MLflow
2. Tune hyperparameters for optimal performance  
3. Evaluate models using appropriate metrics (ROC-AUC, Precision-Recall)
4. Implement proper cross-validation strategy
""")

# Save processed data for model building
print("\nSaving processed datasets...")

# Save different versions for experimentation
datasets_to_save = {
    'original_processed': (X_processed, y),
    'smote_resampled': (X_train_smote, y_train_smote),
    'pca_transformed': (pd.DataFrame(X_train_pca_optimal), y_train_smote),
    'test_set': (X_test, y_test)
}

for name, (X_data, y_data) in datasets_to_save.items():
    X_data.to_csv(f'X_{name}.csv', index=False)
    pd.Series(y_data).to_csv(f'y_{name}.csv', index=False)

print("All datasets saved successfully!")
print("\nNotebook execution completed!")
