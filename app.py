import streamlit as st
import pickle
import pandas as pd

# 🌈 Custom Styling
page_bg = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, #f5f7fa, #c3cfe2, #fbeaff, #fddde6);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: #333333;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

h1, h2, h3, h4 {
    color: #4B0082;
}

.stButton > button {
    background-color: #6a0dad;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5em 1.5em;
    font-weight: 600;
    transition: all 0.3s ease-in-out;
}

.stButton > button:hover {
    background-color: #8a2be2;
    transform: scale(1.05);
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
}

[data-testid="stSidebar"] {
    background-color: #f3e5f5;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# 📦 Load Data and Model
top_skills = pd.read_csv("top_skills.csv", header=None)[0].tolist()

with open("lightgbm_resignation_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_features.pkl", "rb") as f:
    feature_list = pickle.load(f)

# 🎯 Title
st.title("Employee Resignation Predictor")

# 📝 Sidebar Input Form
with st.form("input_form"):
    st.subheader("📋 Enter Employee Details")

    age = st.slider("🧓 Age", 20, 65, 30)
    experience = st.slider("💼 Total Experience (Years)", 0, 40, 5)
    performance_rating = st.slider("📈 Performance Rating (1-5)", 1, 5, 3)
    tenure_years = st.slider("🏢 Tenure at Company (Years)", 0, 20, 3)
    selected_skills = st.multiselect("🛠 Skills", options=top_skills)

    submitted = st.form_submit_button("🔍 Predict Resignation")

if submitted:
    # ✅ Preprocess Input
    skill_score = len(selected_skills) / len(top_skills)
    user_input = pd.DataFrame([{
        "Age": age,
        "Experience (Years)": experience,
        "Performance Rating": performance_rating,
        "Tenure_years": tenure_years,
        "Skill_Score": skill_score
    }])

    user_input_encoded = pd.get_dummies(user_input, drop_first=True)
    user_input_final = user_input_encoded.reindex(columns=feature_list, fill_value=0)

    # 🔮 Predict
    prediction = model.predict(user_input_final)[0]
    probability = round(model.predict_proba(user_input_final)[0][1], 2)

    # 📊 Display Results
    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ The employee is likely to resign.\n\n💡 Probability: {probability}")
    else:
        st.success(f"✅ The employee is likely to stay.\n\n📈 Probability: {1 - probability}")


