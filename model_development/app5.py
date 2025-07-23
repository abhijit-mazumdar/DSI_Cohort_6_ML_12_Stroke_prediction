import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


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
    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
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

        risk_info[feature] = {
            'high_risk_category': high_risk_cat,
            'stroke_rate_high_risk': high_risk_rate,
            'stroke_rate_others': other_rate,
            'risk_increase': lift
        }

    risk_info = {k: v for k, v in risk_info.items() if v['risk_increase'] > 0}
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


def build_model(input_dim):
    model = Sequential()
    model.add(Dense(256, input_shape=(input_dim,)))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(64))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    local_css()
    st.markdown('<h1 class="main-title">Stroke Prediction App</h1>', unsafe_allow_html=True)
    st.markdown("### Please enter your health details below to receive a stroke risk prediction.")

    # Load data
    df = pd.read_csv("/Users/angiean/ML group project/01_materials/stroke_data.csv")
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

    if 'cluster' in df.columns:
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

        # Map user-friendly inputs to dataset labels for consistency
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

        # Map inputs
        mapped_work_type = work_type_map[work_type]
        mapped_smoking_status = smoking_status_map[smoking_status]

        # Prepare user input DataFrame with mapped values
        user_df = pd.DataFrame([{
            'age': age,
            'hypertension': 1 if hypertension == 'Yes' else 0,
            'heart_disease': 1 if heart_disease == 'Yes' else 0,
            'ever_married': 1 if ever_married == 'Yes' else 0,
            'work_type': mapped_work_type,
            'Residence_type': 1 if residence_type == 'Urban' else 0,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': mapped_smoking_status,
            'gender': 0 if gender == 'Male' else 1,
            'stroke': 0  # dummy stroke for alignment
        }])

        # Append user input to df for consistent processing
        df_with_user = pd.concat([df, user_df], ignore_index=True)

        # Recompute risk score for the full dataset including user input
        df_with_user['risk_score'] = df_with_user.apply(lambda row: compute_automated_risk_score(row, risk_info), axis=1)

        # Map segment labels to numeric for modeling
        segment_map = {'Low Risk': 0, 'Moderate Risk': 1, 'Elevated Risk': 2, 'High Risk': 3}
        df_with_user['segment'] = df_with_user['segment'].map(segment_map)
        df_with_user['segment'].fillna(0, inplace=True)  # assign low risk if missing

        # Prepare training and test sets (excluding user input for test)
        train_df = df_with_user.iloc[:-1].copy()
        test_df = df_with_user.iloc[[-1]].copy()

        features_subset = ['age', 'hypertension', 'heart_disease', 'ever_married',
                           'work_type', 'Residence_type', 'avg_glucose_level', 'risk_score', 'segment']

        X = train_df[features_subset]
        y = train_df['stroke']

        numeric_features = ['age', 'avg_glucose_level', 'risk_score']
        categorical_features = ['work_type', 'segment']

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])

        X_train = preprocessor.fit_transform(X)
        y_train = y.values

        X_test = preprocessor.transform(test_df[features_subset])

        # Compute class weights
        cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        cw_dict = {0: cw[0], 1: cw[1]}

        # Build and train neural network
        model = build_model(X_train.shape[1])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]

        with st.spinner("This may take a few moments."):
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=32,
                class_weight=cw_dict,
                callbacks=callbacks,
                verbose=0
            )

        # Predict on test (user input)
        y_prob = model.predict(X_test).flatten()

        # Find best threshold using train data predictions
        train_probs = model.predict(X_train).flatten()
        precision, recall, thresholds = precision_recall_curve(y_train, train_probs)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]

        user_prob = y_prob[0]
        user_pred = 1 if user_prob >= best_threshold else 0

        st.markdown(
            f'<div style="font-size:2.5rem; font-weight:700; color:#212529;">'
            f'Your Stroke Risk Probability: {user_prob:.2%}</div>',
            unsafe_allow_html=True)
        if user_pred == 1:
            st.error("⚠️ High risk detected. Please consult a healthcare professional.")
        # Removed success message as requested

    # Chatbot section
    st.markdown("---")
    st.markdown('<div class="section-container"><h2>Stroke Information Chatbot</h2></div>', unsafe_allow_html=True)
    user_msg = st.text_input("Ask about stroke or risk factors:")
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
    q2 = st.radio("Is smoking a risk factor?", ["Yes", "No"])
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