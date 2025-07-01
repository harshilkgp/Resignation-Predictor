# Resignation Predictor

A user-friendly Streamlit web app that predicts whether an employee is likely to resign, using a LightGBM machine learning model trained on structured HR data. The app provides actionable insights for HR teams and managers to identify at-risk employees and take proactive measures.

---

## 🚀 Features
- **Predicts resignation risk** using:
  - Age
  - Total Experience (Years)
  - Performance Rating
  - Company Tenure (Years)
  - Skill set (via skill score based on top 20 skills)
- **Interactive UI**: Clean, modern, and responsive interface built with Streamlit
- **One-click deployment**: Easily deployable on Streamlit Cloud
- **Handles imbalanced data**: Model trained using SMOTE
- **Skill matching**: Calculates a Skill Score based on the top 20 most in-demand skills

---

## 🌐 Live Demo
Try the app live: [Resignation Predictor Streamlit App](https://resignation-predictor-if2qbxpwgh5piruj4rwips.streamlit.app/)

---

## 📊 Data & Model
- **Dataset**: HR data with columns such as Age, Experience, Performance Rating, Tenure, Skills, and Resignation status.
- **Model**: LightGBM classifier trained on one-hot encoded features, with SMOTE for balancing classes.
- **Skill Score**: Computed as the proportion of top 20 skills possessed by the employee.
- **Key files**:
  - `final merged data.csv`: The main dataset
  - `lightgbm_resignation_model.pkl`: Trained model
  - `model_features.pkl`: List of features used by the model
  - `top_skills.csv`: Top 20 skills used for skill scoring

---

## 🛠️ Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/harshilkgp/Resignation-Predictor.git
   cd Resignation-Predictor/Streamlit app
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage
1. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```
2. **Open your browser** and go to the local URL provided by Streamlit (usually http://localhost:8501)
3. **Input employee details** in the sidebar:
   - Age
   - Total Experience (Years)
   - Performance Rating (1-5)
   - Tenure at Company (Years)
   - Select skills from the top 20 list
4. **Click 'Predict Resignation'** to see the prediction and probability.

---

## 📁 File Structure
- `app.py` : Main Streamlit app
- `final merged data.csv` : HR dataset
- `lightgbm_resignation_model.pkl` : Trained LightGBM model
- `model_features.pkl` : Model input features
- `top_skills.csv` : Top 20 skills for skill scoring
- `train_lightgbm_model.py` : Script to train the model
- `requirements.txt` : Python dependencies

---

## 🙏 Acknowledgements
- Built with [Streamlit](https://streamlit.io/), [LightGBM](https://lightgbm.readthedocs.io/), [scikit-learn](https://scikit-learn.org/), and [imbalanced-learn](https://imbalanced-learn.org/)
- Inspired by real-world HR analytics challenges

---

For questions or suggestions, open an issue or reach out via GitHub.
