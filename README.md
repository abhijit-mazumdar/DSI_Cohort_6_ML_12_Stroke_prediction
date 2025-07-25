# Stroke Prediction with Machine Learning


## Table of Contents

1. [Introduction & Dataset Overview](#1-introduction--dataset-overview)  
2. [Data Cleaning and Preprocessing](#2-data-cleaning-and-preprocessing)  
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)  
4. [Feature Engineering](#4-feature-engineering)  
5. [Model Selection and Development](#5-model-selection-and-development)  
6. [Handling Class Imbalance](#6-handling-class-imbalance)  
7. [Model Evaluation](#7-model-evaluation)
8. [Final Model](#8-final-model)
9. [Interpretability and Explainability](#9-interpretability-and-explainability)  
10. [Deployment](#10-deployment)  
11. [Challenges and Lessons Learned](#11-challenges-and-lessons-learned)  


---

## 1. Introduction & Dataset Overview

### Background and Motivation

Stroke remains one of the foremost causes of death and long-term disability worldwide. It occurs when the blood supply to the brain is interrupted or reduced, leading to brain cell death due to lack of oxygen and nutrients. According to the World Health Organization, approximately 15 million people suffer a stroke each year, with nearly 5 million deaths and 5 million survivors experiencing permanent disabilities.

This project explores the application of ML techniques to predict stroke occurrence using clinical and demographic data from the Kaggle Stroke Prediction Dataset. The goal is to build an accurate, interpretable, and deployable model that can aid healthcare providers and patients alike in stroke risk assessment.



### Dataset Description

The dataset used in this study is publicly available on Kaggle and contains records for 5,110 patients, each described by 12 features, including the binary target variable indicating whether the patient had a stroke.

The features are summarized below:

| Feature           | Description                                                    |
|-------------------|----------------------------------------------------------------|
| **id**            | Unique patient identifier                                      |
| **gender**        | Patient’s gender: Male, Female, or Other                       |
| **age**           | Patient age in years                                           |
| **hypertension**  | Binary indicator: 1 if patient has hypertension, else 0       |
| **heart_disease** | Binary indicator: 1 if patient has heart disease, else 0      |
| **ever_married**  | Marital status: Yes or No                                      |
| **work_type**     | Patient’s work type: Govt_job, Private, Self-employed, Children, Never_worked |
| **Residence_type**| Urban or Rural residence                                       |
| **avg_glucose_level** | Average blood glucose level                                 |
| **bmi**           | Body Mass Index (some values missing)                         |
| **smoking_status**| Smoking status: formerly smoked, never smoked, smokes, unknown |
| **stroke**        | Target variable: 1 if stroke occurred, 0 otherwise            |


### Initial Observations and Challenges

The dataset presents several important characteristics and challenges that shaped the modeling approach:

#### Imbalanced Target Classes

- Stroke cases represent approximately 4.9% (249 out of 5,110) of the dataset, making the classification task highly imbalanced. This imbalance could cause models to be biased towards predicting the majority “no stroke” class if unaddressed.

#### Missing Values

- The **bmi** feature contains missing values for about 4% of the records (201 out of 5,110). Since BMI is a crucial health indicator associated with stroke risk, effective imputation methods are necessary to avoid bias and loss of data.

#### Categorical Variables

- Several features such as gender, work_type, smoking_status, and residence_type are categorical and require encoding into numerical representations that machine learning algorithms can process.

#### Outliers and Noisy Data

- Certain observations show extreme or inconsistent values, particularly in BMI and glucose levels. These outliers need to be carefully handled during preprocessing to prevent distortion of model learning.

#### Moderate Dataset Size

- With just over 5,000 samples, the dataset is sufficient to train classical machine learning models and small neural networks but not large-scale deep learning models.



### Project Objectives

Given these characteristics, the project is designed to accomplish the following:

- **Develop predictive models** capable of accurately classifying patients’ stroke risk, prioritizing recall to minimize missed stroke cases.
- **Effectively handle missing and noisy data** through rigorous preprocessing and imputation strategies.
- **Apply feature engineering** to capture non-linear relationships and interactions that may improve predictive performance.
- **Address class imbalance** using resampling methods, class weighting, and threshold tuning.
- **Provide model interpretability** through tools like SHAP to foster clinical trust and usability.
- **Deploy the model** in an accessible interactive application for end users including clinicians and patients.

---


## 2. Data Cleaning and Preprocessing

### Introduction

Data preprocessing is a critical step in any machine learning project. The quality and format of the input data greatly influence the performance and reliability of predictive models. In this project, preprocessing involved handling missing data, addressing outliers, encoding categorical variables, and scaling numerical features.

The raw stroke prediction dataset contained several imperfections, including missing BMI values, inconsistent categorical entries, and extreme outliers. This section describes the methods applied to clean and transform the data into a machine-learning-friendly format.


### Missing BMI Values



### Importance of BMI in Stroke Prediction

Body Mass Index (BMI) is a crucial indicator of obesity, a known risk factor for stroke. However, in the stroke dataset used, many BMI values are missing. Since accurate BMI data is essential for reliable risk modeling, we must estimate these missing values effectively to maintain dataset integrity and improve prediction outcomes.


### Correlation Analysis to Guide Imputation

To select effective imputation methods, we analyzed the correlation of BMI with two clinically relevant features:

| Correlation Metric           | Age vs BMI | Glucose vs BMI |
|------------------------------|------------|----------------|
| Original Data (with missing) | 0.3334     | 0.1755         |
| After Mean Imputation        | 0.3259     | 0.1688         |
| After Median Imputation      | 0.3243     | 0.1669         |
| After Regression Imputation  | 0.3353     | 0.1836         |

- A moderate positive correlation (~0.33) between Age and BMI confirms age is a meaningful predictor.
- Glucose levels show a weaker but non-negligible relationship with BMI (~0.17–0.18).
- Polynomial regression imputation preserves and slightly improves these relationships better than simpler methods.


## BMI Distribution Comparison

The following plot compares the BMI distributions from the original and imputed datasets:

![BMI Distribution Comparison](Images/BMI.png)

- All methods aim to maintain the original distribution.
- **Regression-based imputation** closely tracks the original curve while slightly reducing distortion compared to mean or median methods.

## Imputation Methods Evaluated

- **Mean Imputation:** Replaces missing BMI with the dataset’s average.
- **Median Imputation:** Uses the median BMI value, offering robustness to outliers.
- **Polynomial Regression Imputation:** Predicts missing BMI using a regression model based on Age and Glucose, capturing nonlinear dependencies.

## Evaluation Metrics

We assessed the imputation approaches on:

- **Predictive Power for Stroke Detection:** Evaluated by AUC and F1 Score.
- **BMI Imputation Accuracy:** Evaluated by Mean Squared Error (MSE) on artificially masked BMI values.

| Imputation Method | Stroke Prediction AUC | Stroke Prediction F1 Score | BMI Imputation MSE (masked data) |
|-------------------|------------------------|-----------------------------|----------------------------------|
| Mean              | 0.7932                 | 0.0000                      | 73.0823                          |
| Median            | 0.7889                 | 0.0000                      | 73.5383                          |
| Regression        | 0.7610                 | 0.0250                      | 59.4719                          |


## Interpretation of Results

**Mean and Median Imputation:**

- Preserve overall dataset structure but underperform in stroke classification (F1 = 0).
- Higher MSE indicates poor BMI estimation accuracy.

**Polynomial Regression Imputation:**

- **Lowest MSE**, indicating more realistic BMI predictions.
- Slight drop in AUC but **non-zero F1 score**, signaling improved stroke case detection.
- Best preserves correlations and real-world BMI behavior.


## Summary and Rationale

- BMI correlates moderately with Age and modestly with Glucose, providing a foundation for imputation.
- **Regression imputation** effectively models these relationships, outperforming simpler methods on accuracy and clinical relevance.
- Better BMI estimation leads to improved data quality, essential for complex model training and interpretation.


### Outlier Detection and Column Removal

Certain extreme values in `bmi` and `avg_glucose_level` were identified as outliers that could skew model training:

- BMI values greater than 70 were considered implausible for the population.
- Glucose levels above 270 mg/dL, which are physiologically extreme, were examined.
- The `gender` column contained a small number of records labeled as `"Other"`. Due to their scarcity and unclear definition, these records were removed.
- The `work_type` category `Children` applies to individuals under 18 years old. For modeling consistency, all `Children` entries were grouped under the category `Never_worked`. This consolidation simplifies the work type variable and aligns with logical age restrictions.
- The `id` column was dropped as it does not contribute to prediction.

---

## 3 Exploratory Data Analysis

Exploratory Data Analysis (EDA) is a fundamental step to understand the characteristics of the dataset and to identify relationships between variables that will inform feature engineering and model development.

### Target Variable Distribution and Class Imbalance

The dataset exhibits a strong class imbalance:

| Stroke Status | Count | Percentage |
|---------------|-------|------------|
| No Stroke (0) | 4,861 | 95.1%      |
| Stroke (1)    | 249   | 4.9%       |

This imbalance poses a risk that naive models may simply predict the majority class, leading to misleadingly high accuracy but poor recall for stroke cases. Therefore, metrics like F1-score, precision, and recall must be emphasized in model evaluation.

### Analysis of Key Features

![Features Graph](Images/stroke_feature_comparison.png)


### Age Distribution

- **Observation**:
  - Non-stroke individuals are evenly distributed across all age groups from 20 to 80.
  - Stroke cases are concentrated in older age groups, especially **ages 50 and above**.

- **Insight**:
  - **Age is a strong predictor** of stroke. 
  - The likelihood of stroke increases significantly with age.


### BMI Distribution

- **Observation**:
  - BMI for both groups follows a **normal distribution** centered around **25–30**.
  - A slight increase in stroke cases is observed at higher BMI levels.

- **Insight**:
  - **BMI has a mild correlation** with stroke risk.
  - It may not be a standalone predictor but can enhance model performance when combined with other features.


### Average Glucose Level Distribution

- **Observation**:
  - The distribution is **right-skewed**, with most individuals having glucose levels between **80–150 mg/dL**.
  - Stroke cases are noticeably more frequent in individuals with **glucose levels above 150 mg/dL**.

- **Insight**:
  - **High glucose levels**, potentially indicating diabetes or prediabetes, show a **strong association** with stroke.
  - This feature is likely **highly important** in prediction models.

### Categorical Variable Analysis

- **Gender:** After excluding rare 'Other' category, stroke prevalence was similar across male and female patients.
- **Hypertension:** Strong positive association with stroke; hypertensive patients had significantly higher stroke incidence.
- **Heart Disease:** Similarly, heart disease presence correlated strongly with stroke.
- **Smoking Status:** Current and former smokers had higher stroke rates compared to those who have never smoked before.
- **Work Type and Residence Type:** No substantial difference in stroke prevalence detected.

### Feature Interaction Analysis

- Investigated interactions between features, particularly between continuous variables.
- The product term `age * bmi` showed a moderate correlation with stroke, indicating that the combined effect of aging and BMI might be relevant.
- Polynomial terms such as `age²` and `glucose²` captured observed non-linear effects.
- Visualizations such as scatter plots and boxplots were used to inspect these interactions.

### Correlation Matrix

- Calculated Pearson correlation coefficients for numerical features and the target.
- `age` and `avg_glucose_level` showed positive correlations with stroke.
- `bmi` showed a weak positive correlation.
- Checked feature-to-feature correlations to detect multicollinearity risks.

### Feature Binning

- Created bins for continuous variables to discretize them into meaningful groups:
  - **Age:** Divided into <40 years, 40–60 years, and >60 years categories to align with clinical risk brackets.
  - **Avg Glucose Level:** Grouped into normal (<140 mg/dL), prediabetic (140–199 mg/dL), and diabetic (≥200 mg/dL) ranges.
- Binning helped simplify complex relationships and was useful for models that benefit from categorical inputs.


## Feature Importance Analysis

To better understand which features contribute most to predicting stroke, we trained a Random Forest classifier with class weighting to handle imbalance and extracted feature importances.

The top 20 features ranked by their importance in the model are shown below:

![Feature Importances from Random Forest](Images/feature_importance.png)

### Insights from Feature Importance

- **Age** emerged as the most important predictor, consistent with medical knowledge that stroke risk increases with age.
- **Average glucose level** was also highly predictive, reflecting the impact of blood sugar on stroke risk.
- Medical history indicators such as **hypertension** and **heart disease** showed strong contributions.

This analysis informed further feature engineering and model tuning efforts.

## Stroke Probability by BMI, Glucose Level, and Age

To explore non-linear relationships between numerical features and stroke risk, we binned and plotted stroke probability across three key continuous variables:

- **BMI (Body Mass Index)**
- **Average Glucose Level**
- **Age**

Each variable was divided into 10 equal-width bins. For each bin, we calculated the percentage of patients who experienced a stroke. This allowed us to visualize how stroke risk varies across different ranges of these features.

### What the Plots Show

![Stroke Probability Trends](Images/stroke_probability.png)

## 1. Stroke Probability vs **BMI**

- Stroke probability **increases with BMI** and **peaks around 32**, suggesting elevated risk in individuals who are moderately overweight.
- After the peak, stroke probability **declines sharply**, reaching **0% at BMI values above 66** in the dataset.
- The trend is **non-linear and unimodal**, indicating that BMI should **not** be treated as a purely linear feature.

---

## 2. Stroke Probability vs **Average Glucose Level**

- Stroke probability remains relatively stable and low **up to ~170 mg/dL**.
- Beyond that point, there is a **sharp increase**, especially after **200 mg/dL**, with stroke probability reaching **20% at 260 mg/dL**.
- This suggests strong correlation between **hyperglycemia (high blood sugar)** and stroke risk.


---

## 3. Stroke Probability vs **Age**

- Stroke probability is minimal at younger ages, begins to rise gradually after **age 40**, and **increases steeply after age 60**.
- Peaks at **20% by age 78**, showing a **clear exponential trend**.
- Implications that age has the **strongest monotonic and exponential relationship** with stroke probability.


### Why This Matters

Understanding how stroke probability changes across these features helps in:

- **Feature engineering**: We can use this information to transform variables (e.g., create bins or quadratic terms) that better capture non-linear risk patterns.
- **Model interpretability**: These trends give clinicians and stakeholders a visual understanding of risk factors, even without black-box models.
- **Threshold-based interventions**: Health policies or apps might set alerts when glucose levels exceed ~180 or BMI crosses 35, based on real observed risk jumps.
  

---
## 4 feature engineering


## Risk Score Summary


The **Stroke Risk Score** is a composite feature designed to quantify a person’s likelihood of experiencing a stroke. It combines multiple individual risk factors into a single numerical value to support early detection, risk stratification, and targeted intervention.



## How the Risk Score Is Built

1. **Identify Key Risk Factors**  
   We use features like hypertension, heart disease, marital status, gender, work type, smoking status, and residence type.

2. **Determine High-Risk Groups**  
   For each feature, we find the category (e.g., "Yes", "Male", "Self-employed") with the **highest stroke rate**. This becomes the "high-risk" category.

3. **Calculate Risk Increase**  
   We measure how much higher the stroke rate is for the high-risk group compared to others.

4. **Assign Weights**  
   Each feature is given a **weight** based on its risk increase, normalized so that the highest-impact factor gets a weight of `1.0`.

5. **Compute Individual Scores**  
   Each person gets a score by summing the weights of the risk factors they belong to.  
   > Example: If a person has heart disease and hypertension, their risk score is the sum of both weights.

---

## Risk Factors and Assigned Weights

| Risk Factor      | High-Risk Category Value | Stroke Rate (High Risk) | Stroke Rate (Others) | Risk Increase (Difference) | Relative Weight |
|------------------|--------------------------|------------------------|---------------------|----------------------------|-----------------|
| **Hypertension** | 1                        | 13.25%                 | 3.97%               | 9.28%                      | 0.72            |
| **Heart Disease**| 1                        | 17.03%                 | 4.18%               | 12.85%                     | 1.00            |
| **Ever Married** | Yes                      | 6.56%                  | 1.65%               | 4.91%                      | 0.38            |
| **Gender**       | Male                     | 5.11%                  | 4.71%               | 0.40%                      | 0.03            |
| **Work Type**    | Self-employed            | 7.94%                  | 4.29%               | 3.65%                      | 0.28            |
| **Smoking Status**| Formerly Smoked          | 7.92%                  | 4.24%               | 3.68%                      | 0.29            |
| **Residence Type**| Urban                    | 5.20%                  | 4.53%               | 0.67%                      | 0.05            |


### Explanation of Terms

- **Stroke Rate (High-Risk):** % of people with the risk factor who had a stroke.
- **Stroke Rate (Others):** % of people without the risk factor who had a stroke.
- **Risk Increase:** Difference between high-risk and others.
- **Weight:** Relative importance of each factor (scaled so that the highest = 1.00).


## Clustering for Risk Segmentation


To improve stroke prediction and better understand risk profiles, we applied **K-Means Clustering** to segment the population into distinct health risk groups.


- **Risk Stratification:** Group individuals by similar health and risk factors to identify high-risk segments.
- **Feature Engineering:** Use cluster labels as features in predictive models.
- **Interpretability:** Gain insights into characteristics of different risk groups.
- **Personalization:** Enable targeted interventions based on risk segment.

### Methodology

- **Features Used:** Age, average glucose level, BMI, hypertension, and heart disease status.
- **Preprocessing:** Missing BMI values were imputed using polynomial regression, and numeric features were standardized.
- **Cluster Selection:** Tested cluster counts (k) from 2 to 10 using Silhouette scores.
- **Optimal Number of Clusters:** 3 (silhouette score = 0.2458).

### Cluster Summary

| Cluster | Stroke Rate | Avg Risk Score | Age (years) | Avg Glucose Level | BMI    | Hypertension Rate | Heart Disease Rate | Risk Label     |
|---------|-------------|----------------|-------------|-------------------|--------|-------------------|--------------------|----------------|
| 0       | 0.13%       | 0.10           | 16.93       | 94.09             | 23.00  | 0.00%             | 0.06%              | Low Risk       |
| 1       | 14.29%      | 0.92           | 61.81       | 205.67            | 33.69  | 27.83%            | 17.56%             | High Risk      |
| 2       | 5.29%       | 0.64           | 53.47       | 89.40             | 31.11  | 10.90%            | 5.50%              | Elevated Risk  |

### Interpretation

- **High Risk (Cluster 1):** Older individuals with high glucose and BMI, and higher prevalence of hypertension and heart disease, showing the highest stroke rate.
- **Elevated Risk (Cluster 0):** Middle-aged group with moderate glucose, BMI, and comorbidities, with moderate stroke risk.
- **Low Risk (Cluster 2):** Younger, healthier individuals with very low stroke incidence.

### Conclusion

Clustering effectively identified meaningful subgroups within the population. Incorporating these risk segments enhances predictive modeling by providing additional context beyond individual features, improving both model interpretability and performance in stroke prediction.


### Visualizing Clusters with UMAP

Below is a UMAP projection visualizing the clusters in 2D space, showing clear separation among risk groups:

![UMAP Projection of Stroke Risk Clusters](Images/clustering.png)

---

## 5. Model Selection and Development


To identify the best predictive model for stroke detection, we evaluated 32 classifiers across diverse algorithm families, including:

- **Ensemble methods:** RandomForest, GradientBoosting, AdaBoost, Bagging, ExtraTrees, HistGradientBoosting  
- **Linear models:** LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressive, Perceptron  
- **Naive Bayes variants:** GaussianNB, BernoulliNB, ComplementNB, MultinomialNB, CategoricalNB  
- **Nearest neighbor methods:** KNeighbors, RadiusNeighbors, NearestCentroid  
- **Support Vector Machines:** SVC, LinearSVC, NuSVC  
- **Tree-based models:** DecisionTree, ExtraTree  
- **Discriminant analysis:** LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis  
- **Semi-supervised learning:** LabelPropagation, LabelSpreading  
- **Others:** CalibratedClassifierCV, DummyClassifier, OneVsRest with LogisticRegression and SVC

Each classifier was wrapped in a preprocessing pipeline to standardize numeric features (using mean imputation and scaling) and encode categorical variables (most frequent imputation plus one-hot encoding), ensuring consistent, leak-proof training.

The dataset was stratified into training and testing subsets to preserve class distribution. Models were trained with default hyperparameters to establish baseline results.

An ensemble model was built combining four strong classifiers with balanced class weights:  
- Logistic Regression  
- Random Forest (200 trees)  
- Gradient Boosting (200 estimators)  
- AdaBoost (200 estimators)  

This ensemble employs **soft voting**, averaging predicted probabilities from each model for robust final predictions.


### Neural Network Model Implemented

### Architecture and Training

- Built a deep neural network with multiple dense layers using ReLU activation, batch normalization, and dropout for regularization.
- The output layer uses sigmoid activation for binary stroke classification.
- Class weights were applied to address data imbalance during training.
- Optimized with binary cross-entropy loss and Adam optimizer.
- Early stopping and learning rate reduction on plateau were used to prevent overfitting and improve convergence.


This comprehensive exploration and modular pipeline design allow for systematic experimentation and robust stroke risk modeling.

---

## 6. Handling Class Imbalance


### Data-Level Techniques

### Resampling Methods
- **Random Oversampling:** Duplicate minority class samples to balance the dataset.  
- **Random Undersampling:** Remove samples from the majority class to balance classes.  
- **Synthetic Oversampling:** Generate new synthetic minority samples using:  
  - SMOTE (Synthetic Minority Oversampling Technique)  
  - Borderline-SMOTE  
  - ADASYN (Adaptive Synthetic Sampling)  
  - SMOTE-ENN / SMOTE-Tomek Links (SMOTE combined with data cleaning)  
- **Cluster-Based Oversampling:** Generate synthetic samples within minority class clusters.  


### Algorithm-Level Techniques

- **Ensemble Methods for Imbalance:**  
  - Balanced Random Forest  
  - EasyEnsemble and BalanceCascade  
  - RUSBoost and SMOTEBoost  
- **One-Class Classification / Anomaly Detection:** Model majority class and detect minority as anomalies.  
- **Custom Loss Functions:** Use weighted loss or focal loss in neural networks to emphasize minority class.


### Evaluation and Validation Techniques

- Use imbalance-sensitive metrics:  
  - Precision, Recall, F1-score (focus on minority class)  
  - ROC AUC, Precision-Recall AUC  
  - Matthews Correlation Coefficient (MCC)  
  - Cohen’s Kappa  
  - Balanced Accuracy  
- Use stratified train/test splits and cross-validation to preserve class ratios.  
- Analyze confusion matrices for detailed error insights.

---



## 7. Model Evaluation


## Evaluation Approach

Models were evaluated using key classification metrics:

- **Precision** – how well the model avoids false alarms
- **Recall** – how well the model identifies true stroke cases (especially important)
- **F1 Score** – a balance of precision and recall
- **Support** – how many actual cases were in each class

Because stroke is rare in the dataset, most models scored well on overall accuracy but struggled to detect stroke cases accurately. This highlights the importance of focusing on performance for the minority "Stroke" class.


## Stroke Detection Performance


### Overview

Multiple classifiers were trained and evaluated on the stroke prediction task using a highly imbalanced dataset (majority "No Stroke" cases). Due to this imbalance, overall accuracy was not sufficient to measure performance. Instead, focus was placed on precision, recall, and F1-score for the minority "Stroke" class.

### General Results

| Metric                         | Range         |
|--------------------------------|---------------|
| Overall Accuracy               | 70% – 95%     |
| Stroke Recall (Sensitivity)    | 8% – 82%      |
| Stroke Precision               | < 25%         |
| Stroke F1 Score                | < 0.25        |
| Weighted Average F1            | ~0.80 – 0.93  |
| Macro Average F1               | ~0.50 – 0.60  |

- High accuracy was mainly due to correct classification of the majority "No Stroke" cases.
- Stroke recall and precision varied widely but were generally low, indicating challenges in detecting true stroke cases while avoiding false positives.
- Some classifiers failed to train due to data incompatibilities.

### Stroke Class Performance by Classifier

| Model                | Precision | Recall | F1 Score | Notes                              |
|----------------------|-----------|--------|----------|----------------------------------|
| AdaBoost             | 0.12      | 0.82   | 0.21     | Very high recall, very low precision |
| SGDClassifier        | 0.18      | 0.76   | 0.28     | Best balance among linear models |
| LogisticRegression   | 0.14      | 0.80   | 0.24     | High recall, moderate precision  |
| RandomForest         | 0.15      | 0.20   | 0.17     | Low recall despite decent precision |
| GradientBoosting     | 0.17      | 0.60   | 0.27     | Balanced recall and precision    |
| PassiveAggressive    | 0.16      | 0.46   | 0.24     | Moderate recall, low precision   |
| GaussianNB           | 0.08      | 0.94   | 0.16     | Very high recall, very low precision |
| KNeighbors           | 0.11      | 0.38   | 0.17     | Low recall and precision         |
| DummyClassifier      | 0.00      | 0.00   | 0.00     | No stroke detection              |

### Key Takeaways

- Models generally classify the majority "No Stroke" class well but struggle with accurate stroke detection.
- Linear models achieve higher recall but at the cost of many false positives (low precision).
- Tree-based and ensemble models have mixed results with moderate recall but low precision.
- Improving stroke precision and recall remains a critical challenge.
- Future work should focus on advanced imbalance handling, threshold tuning, and specialized modeling techniques for better minority class detection.

## Neural Network Model Results

### Architecture and Training

- Neural network with dense layers, ReLU activation, dropout, and sigmoid output
- Used class weighting to handle imbalanced data
- Optimized using binary cross-entropy and Adam
- Early stopping and learning rate scheduling applied

### Evaluation Metrics

| Metric                    | Value   |
|---------------------------|---------|
| Best F1 Threshold         | 0.753   |
| F1 Score (Stroke class)   | 0.3429  |
| ROC AUC                   | 0.8161  |
| Accuracy                  | 0.89    |

### Classification Report

| Class      | Precision | Recall | F1 Score | Support |
|------------|-----------|--------|----------|---------|
| No Stroke  | 0.98      | 0.91   | 0.94     | 972     |
| Stroke     | 0.25      | 0.58   | 0.35     | 50      |

- Improved recall over baseline models, capturing more true stroke cases
- Lower precision, meaning some false positives
- AUC of 0.8161 shows strong separation ability between classes


## Top Feature Combinations (Neural Network F1 Score)

From 100 randomized trials of feature subset combinations, the following produced the highest F1 scores:

| Rank | Features                                                                                             | F1 Score | ROC AUC |
|------|------------------------------------------------------------------------------------------------------|----------|---------|
| 1    | age, hypertension, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status   | 0.4091   | 0.8444  |
| 2    | gender, age, hypertension, heart_disease, ever_married, work_type, avg_glucose_level, bmi, smoking_status, segment | 0.3977 | 0.8522  |
| 3    | gender, age, heart_disease, work_type, Residence_type, bmi, segment                                  | 0.3735   | 0.8479  |
| 4    | gender, age, hypertension, work_type, avg_glucose_level, bmi, smoking_status, segment                | 0.3681   | 0.8434  |
| 5    | age, hypertension, ever_married, work_type, Residence_type, avg_glucose_level, bmi, risk_score       | 0.3646   | 0.8327  |
| 6    | age, hypertension, heart_disease, Residence_type, avg_glucose_level, bmi, smoking_status, risk_score | 0.3646   | 0.8362  |
| 7    | age, hypertension, ever_married, bmi, smoking_status, segment                                        | 0.3626   | 0.8403  |
| 8    | gender, age, hypertension, work_type, bmi, segment                                                   | 0.3613   | 0.8458  |
| 9    | gender, age, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, risk_score, segment | 0.3611 | 0.8313  |
| 10   | gender, age, bmi, smoking_status, risk_score, segment                                                | 0.3586   | 0.8334  |

### Interpretation

- **Age** appears in every top combination — the strongest predictor.
- **Glucose level**, **BMI**, **hypertension**, and **risk_score** are consistently impactful.
- Custom features like **risk_score** and **segment** boosted model performance.
- Best F1 scores range from **0.35 to 0.41**, while AUC values remain high (~0.83–0.85).


## Summary and Recommendations

- Most models perform well on "No Stroke" detection but struggle with minority class prediction.
- Neural networks with engineered features and threshold tuning provided the best stroke recall.
- Key predictors include age, hypertension, glucose level, BMI, and smoking status.
- Future improvements could include ensemble learning, cost-sensitive models, and more robust sampling techniques.

---

## 8. Final Model 

We selected this SGDClassifier model because it strikes an effective balance between correctly identifying stroke cases and maintaining overall predictive performance, which is critical in medical diagnosis.

### Key Performance Metrics

- **Best threshold for F1 score:** 0.605  
- **F1 Score (stroke):** 0.3396  
- **AUC Score:** 0.8474  

### Detailed Classification Performance

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| No Stroke   | 0.98      | 0.87   | 0.92     | 972     |
| Stroke      | 0.22      | 0.72   | 0.34     | 50      |

- **Overall accuracy:** 86%  
- **Macro average F1 score:** 0.63  
- **Weighted average F1 score:** 0.90  

### Why This Model?

- **High AUC (0.8474):** This indicates the model has strong ability to distinguish between stroke and non-stroke cases overall.

- **Emphasis on Recall for Stroke (72%):** Prioritizing recall is vital in stroke prediction because missing true stroke cases (false negatives) can have severe health consequences. The model correctly identifies most stroke patients.

- **Acceptable Precision Trade-off (22%):** While precision is lower due to class imbalance, it is acceptable in this screening context where detecting as many true stroke cases as possible is more important than avoiding false positives.

- **Balanced F1 Score:** The stroke class F1 score reflects a good balance between precision and recall given the imbalanced dataset.

- **Strong Performance on No-Stroke Class:** High precision and recall for non-stroke cases ensure the model minimizes unnecessary alerts.

### Summary

This model was chosen for its strong discrimination power and critical ability to detect the minority class (stroke) effectively, aligning with the goal of early stroke detection and timely intervention.

---

## 9. Interpretability and Explainability

While model accuracy and metrics are important, explainability is critical in healthcare applications such as stroke prediction.

Although not covered explicitly in this implementation, the next steps involve applying interpretability methods such as:

- **SHAP (SHapley Additive exPlanations):** To quantify feature contributions for individual predictions  
- **Feature importance analysis:** To identify the most influential predictors  
- **Partial dependence plots:** To visualize the effect of specific features on model output  

These tools will aid in understanding how models make decisions, increasing clinical trust and facilitating actionable insights for practitioners.


To interpret the model’s decisions, we used **SHAP (SHapley Additive exPlanations)** via KernelExplainer on the trained neural network.

![SHAP Summary Plot](Images/shap.png)

- **Age** was found to be the **most important feature** driving stroke predictions.
- Other influential features included `work_type` and `segment`
- This aligns with medical understanding that age is a strong risk factor for stroke.

---

## 10. Deployment

## Stroke Prediction Streamlit App

This project includes a user-friendly Streamlit web application for predicting stroke risk based on user health inputs.

## Features

### User Health Input

- Age, Gender, Hypertension, Heart Disease  
- Marital Status, Work Type, Residence Type  
- Average Glucose Level, BMI, Smoking Status  



### Data Preprocessing & Modeling

- **BMI Imputation**: Missing BMI values are imputed using **polynomial regression** based on age and glucose level.
- **Categorical Encoding**: One-hot encoding for features like work type and smoking status.
- **Risk Scoring**: Automatically assigns a **risk score** based on high-risk categories in categorical variables.
- **Segmentation**: Clustering (KMeans + silhouette analysis) segments users into:
  - Low Risk, Moderate Risk, Elevated Risk, High Risk
- **Final Classifier**:  
  - **SGDClassifier** (modified_huber loss, balanced class weights)
  - Trained on preprocessed features + risk score + cluster segment
  - **Threshold tuning** for optimal F1-score


### Prediction Output

- Outputs **stroke risk probability** and **label** (High/Low risk)
- Color-coded display (green for low, red for high)
- High risk triggers an **alert to consult a healthcare professional**


- **Interactive Sections**  
  - **Stroke Information Chatbot:** Answers questions about stroke symptoms, risk factors, and emergency response (FAST).  
  - **Stroke Quiz:** A short quiz to increase user awareness of stroke facts.  
  - **Habit Tracker:** Tracks healthy habits with a gamified points system and progress bar.
 
---
    
## 11. Challenges and lessons learned

### Challenges

- **Class Imbalance:**  
  The dataset had a large imbalance between stroke and non-stroke cases, making it difficult for the model to accurately predict minority class events.

- **Missing Data:**  
  Missing values in key numeric features like BMI required careful imputation to maintain data integrity and prevent bias.

- **Feature Engineering:**  
  Managing categorical variables with multiple categories necessitated proper encoding (one-hot encoding) and seamless integration into the preprocessing pipeline.

- **Model Selection and Tuning:**  
  Balancing interpretability and predictive performance was challenging. Logistic regression was selected, with class weighting and threshold tuning to improve results.

- **Explainability:**  
  Incorporating SHAP values helped interpret model predictions but added complexity to the workflow.

- **User Interface Design:**  
  Creating an engaging, educational, and easy-to-use app with features like a chatbot, quiz, and habit tracker required thoughtful design and customization.

### Lessons Learned


- **Data quality matters more than modeling:** Clean, accurate, and well-preprocessed data is essential for building reliable stroke prediction models.

- **Imbalanced classes are very challenging:** Predicting minority class (stroke) accurately remains difficult due to the imbalance in the dataset.

- **Unsupervised methods can help feature engineering:** Techniques like clustering or anomaly detection may provide useful new features for better predictions.

- **More features are needed for accurate stroke prediction:** Stroke risk depends on many factors; limited features restrict model performance.

- **Balancing recall and precision is difficult:** Achieving both high recall and precision for stroke prediction is challenging.

- **Neural networks offer a good balance:** Deep learning models can provide better trade-offs between recall and precision compared to simpler models.
