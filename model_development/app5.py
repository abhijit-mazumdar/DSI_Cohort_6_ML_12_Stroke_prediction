import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def local_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

        html, body, #root, .appview-container {
            background-color: #f8f9fa;
            color: #212529;
            font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', Oxygen,
                Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            font-size: 18px;
            line-height: 1.6;
        }
        .css-18e3th9 {padding: 3rem 5rem 5rem 5rem !important;}
        .stApp > header {
            background-color: #ffffff;
            padding: 1rem 2rem;
            font-size: 2.5rem;
            font-weight: 700;
            color: #212529;
            border-bottom: 1px solid #dee2e6;
        }
        .main-title {
            font-size: 3.5rem !important;
            font-weight: 700;
            text-align: center;
            color: #212529;
            margin-bottom: 1.5rem;
        }
        h1, h2, h3 {
            color: #212529 !important;
            font-weight: 700;
        }
        div.stButton > button {
            background-color: #007bff;
            border-radius: 6px;
            border: none;
            color: white;
            font-size: 1.6rem;
            font-weight: 600;
            padding: 0.75rem 2rem;
            transition: background-color 0.3s ease;
            width: 100%;
        }
        div.stButton > button:hover {
            background-color: #0056b3;
            cursor: pointer;
        }
        input[type="number"], select, .stTextInput > div > input {
            font-size: 1.6rem !important;
            padding: 0.5rem !important;
            border: 1.5px solid #ced4da !important;
            border-radius: 6px !important;
            color: #212529 !important;
            background-color: white !important;
            box-shadow: none !important;
        }
        label {
            font-size: 1.4rem !important;
            font-weight: 600;
            color: #495057 !important;
            margin-bottom: 0.3rem !important;
            display: block;
        }
        .stCheckbox > div, .stRadio > div {
            font-size: 1.4rem !important;
            color: #495057 !important;
            margin-bottom: 1rem;
        }
        .stProgress > div > div > div > div {
            background-color: #007bff !important;
            border-radius: 6px !important;
        }
        .section-container {
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 0 6px rgba(0,0,0,0.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def impute_regression(df):
    train = df[df['bmi'].notna()]
    test = df[df['bmi'].isna()]
    X_train = train[['age', 'avg_glucose_level']]
    y_train = train['bmi']
    X_test = test[['age', 'avg_glucose_level']]
    model = Pipeline([('poly', PolynomialFeatures(2)), ('lr', LinearRegression())])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    df.loc[df['bmi'].isna(), 'bmi'] = preds
    return df


def compute_risk_info(df, categorical_features):
    risk_info = {}
    for feature in categorical_features:
        rates = df.groupby(feature)['stroke'].mean()
        high_risk_cat = rates.idxmax()
        high_risk_rate = rates.max()
        other_rate = df.loc[df[feature] != high_risk_cat, 'stroke'].mean()
        lift = high_risk_rate - other_rate
        if lift > 0:
            risk_info[feature] = {
                'high_risk_category': high_risk_cat,
                'stroke_rate_high_risk': high_risk_rate,
                'stroke_rate_others': other_rate,
                'risk_increase': lift
            }
    max_lift = max(v['risk_increase'] for v in risk_info.values())
    for k in risk_info:
        risk_info[k]['weight'] = risk_info[k]['risk_increase'] / max_lift
    return risk_info


def compute_automated_risk_score(row, risk_info):
    score = 0
    for feature, info in risk_info.items():
        if row[feature] == info['high_risk_category']:
            score += info['weight']
    return score


def main():
    local_css()
    st.markdown('<h1 class="main-title">Stroke Early Prevention App</h1>', unsafe_allow_html=True)
    st.markdown("### Please enter your details below to assess your risk profile.")

    # Load data
    df = pd.read_csv("../data/healthcare-dataset-stroke-data.csv")
    df.drop('id', axis=1, inplace=True)
    df = df[df['gender'] != 'Other']

    # Impute missing BMI using regression
    df = impute_regression(df)

    # Clean categorical mappings for original data only
    df['work_type'] = df['work_type'].replace('children', 'Never_worked')
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})
    df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})

    # Compute risk scores based on categorical stroke rates
    categorical_features = ['work_type', 'smoking_status', 'ever_married', 'Residence_type']
    risk_info = compute_risk_info(df, categorical_features)
    df['risk_score'] = df.apply(lambda row: compute_automated_risk_score(row, risk_info), axis=1)

    # Clustering for segmentation
    num_features = ['age', 'avg_glucose_level', 'bmi', 'risk_score']
    cat_features = ['work_type', 'smoking_status']

    preprocessor_cluster = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    X_cluster = preprocessor_cluster.fit_transform(df)

    # Find best k by silhouette score
    best_k = None
    best_score = -1
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_cluster)
        score = silhouette_score(X_cluster, labels)
        if score > best_score:
            best_score = score
            best_k = k

    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['cluster'] = kmeans_final.fit_predict(X_cluster)

    cluster_stats = df.groupby('cluster').agg({
        'stroke': 'mean',
        'risk_score': 'mean',
        'age': 'mean',
        'avg_glucose_level': 'mean',
        'bmi': 'mean',
        'hypertension': 'mean',
        'heart_disease': 'mean'
    }).rename(columns={'stroke': 'stroke_rate', 'risk_score': 'avg_risk_score'})

    def label_cluster(row):
        if row.stroke_rate < 0.02:
            return 'Low Risk'
        elif row.stroke_rate < 0.05:
            return 'Moderate Risk'
        elif row.stroke_rate < 0.10:
            return 'Elevated Risk'
        else:
            return 'High Risk'

    cluster_stats['risk_label'] = cluster_stats.apply(label_cluster, axis=1)
    label_map = cluster_stats['risk_label'].to_dict()
    df['segment'] = df['cluster'].map(label_map).astype('category')

    df.drop(columns=['cluster'], inplace=True)

    # User Input Section
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 0, 120, 50)
        hypertension = st.selectbox("Hypertension?", ['No', 'Yes'])
        heart_disease = st.selectbox("Heart Disease?", ['No', 'Yes'])
        ever_married = st.selectbox("Ever Married?", ['No', 'Yes'])
        work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Government Job', 'Never worked'])

    with col2:
        residence_type = st.selectbox("Residence Type", ['Urban', 'Rural'])
        avg_glucose_level = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        smoking_status = st.selectbox("Smoking Status", ['formerly smoked', 'never smoked', 'smokes', 'Dont know'])
        gender = st.selectbox("Gender", ['Male', 'Female'])

    if st.button("Prediction"):

        work_type_map = {
            'Private': 'Private',
            'Self-employed': 'Self-employed',
            'Government Job': 'Govt_job',
            'Never worked': 'Never_worked'
        }
        smoking_status_map = {
            'formerly smoked': 'formerly smoked',
            'never smoked': 'never smoked',
            'smokes': 'smokes',
            'Dont know': 'Unknown'
        }

        mapped_work_type = work_type_map[work_type]
        mapped_smoking_status = smoking_status_map[smoking_status]

        user_df = pd.DataFrame([{
            'gender': 0 if gender == 'Male' else 1,
            'age': age,
            'hypertension': 1 if hypertension == 'Yes' else 0,
            'heart_disease': 1 if heart_disease == 'Yes' else 0,
            'ever_married': 1 if ever_married == 'Yes' else 0,
            'work_type': mapped_work_type,
            'Residence_type': 1 if residence_type == 'Urban' else 0,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': mapped_smoking_status,
            'stroke': 0  # dummy for alignment
        }])

        # Append user to original df for consistent preprocessing
        df_with_user = pd.concat([df, user_df], ignore_index=True)

        # Recompute risk score including user input
        df_with_user['risk_score'] = df_with_user.apply(lambda row: compute_automated_risk_score(row, risk_info), axis=1)

        # Map segment labels to numeric
        segment_map = {'Low Risk': 0, 'Moderate Risk': 1, 'Elevated Risk': 2, 'High Risk': 3}
        df_with_user['segment'] = df_with_user['segment'].map(segment_map)
        df_with_user['segment'].fillna(0, inplace=True)  # default low risk if missing

        # Split train and user input for prediction
        train_df = df_with_user.iloc[:-1]
        test_df = df_with_user.iloc[[-1]]

        features = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
                    'smoking_status', 'risk_score', 'segment']
        X_train = train_df[features]
        y_train = train_df['stroke']

        X_test = test_df[features]

        numeric_features = ['age', 'avg_glucose_level', 'bmi', 'risk_score']
        categorical_features = ['gender', 'work_type', 'Residence_type', 'smoking_status', 'segment']

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('sgd', SGDClassifier(
                penalty='l2',
                max_iter=2000,
                loss='modified_huber',
                learning_rate='optimal',
                class_weight='balanced',
                alpha=0.01,
                random_state=42))
        ])

        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Find best threshold on train data for F1
        train_probs = pipeline.predict_proba(X_train)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_train, train_probs)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]

        user_prob = y_prob[0]
        user_pred = int(user_prob >= best_threshold)

        # Show Low Risk or High Risk label with color
        risk_label = "High Risk" if user_pred == 1 else "Low Risk"
        color = "#dc3545" if user_pred == 1 else "#28a745"  # red if high risk, green if low

        st.markdown(
            f'<div style="font-size:2.5rem; font-weight:700; color:{color};">'
            f'Your Stroke Risk: {risk_label}</div>',
            unsafe_allow_html=True
        )
        if user_pred == 1:
            st.error("⚠️ Please consult a healthcare professional.")

    # Chatbot section
    st.markdown("---")
    st.markdown('<div class="section-container"><h2>Simple Stroke Knowledge Chatbot</h2></div>', unsafe_allow_html=True)
    user_msg = st.text_input("Sample Questions: Ask about stroke or risk factors:")
    if user_msg:
        user_msg_lower = user_msg.lower()
        if "stroke" in user_msg_lower:
            st.write("Stroke is caused by an interruption of blood supply to the brain.")
        elif "fast" in user_msg_lower:
            st.write("FAST: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services.")
        elif "symptom" in user_msg_lower:
            st.write("Symptoms include numbness, confusion, trouble seeing, dizziness, and severe headache.")
        elif "risk" in user_msg_lower:
            st.write("Risk factors: hypertension, smoking, obesity, heart disease, and diabetes.")
        else:
            st.write("Please enter questions about stroke symptoms, risks, and prevention.")

    # Quiz section
    st.markdown("---")
    st.markdown('<div class="section-container"><h2>Stroke Quiz</h2></div>', unsafe_allow_html=True)
    q1 = st.radio("What does FAST stand for?", ["Face drooping", "Fast running", "Arm weakness", "Speech difficulty"])
    q2 = st.radio("Is smoking a rißsk factor?", ["Yes", "No"])
    q3 = st.radio("Which organ is affected by stroke?", ["Heart", "Brain", "Liver"])
    if st.button("Submit Quiz"):
        score = 0
        if q1 == "Face drooping": score += 1
        if q2 == "Yes": score += 1
        if q3 == "Brain": score += 1
        st.markdown(f'<h3 style="color:#212529;">Your Score: {score} / 3</h3>', unsafe_allow_html=True)

    # Habit Tracker
    st.markdown("---")
    st.markdown('<div class="section-container"><h2>Habit Tracker</h2></div>', unsafe_allow_html=True)
    water = st.checkbox("Drink Water")
    walk = st.checkbox("Walk 30 mins")
    meds = st.checkbox("Take Medication")
    veg = st.checkbox("Eat Vegetables")
    smoke_avoid = st.checkbox("Avoid Smoking")
    points = sum([water, walk, meds, veg, smoke_avoid])
    st.progress(points / 5)
    st.markdown(f'<p style="font-size:1.3rem; color:#212529;">Points earned: {points}</p>', unsafe_allow_html=True)



if __name__ == "__main__":
    main()
