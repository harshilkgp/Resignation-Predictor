import pandas as pd
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import re

df = pd.read_csv("final merged data.csv")

# Preprocess target and features
df['Resigned'] = df['Resigned (Y/N)'].map({'Y': 1, 'N': 0})
df.drop(columns=['Resigned (Y/N)', 'Full Name', 'Ethnicity', 'Annual Salary'], inplace=True)

skills_tokens = []
for skill_str in df['Skills'].dropna():
    skills_tokens.extend([s.strip().title() for s in re.split(',|;', skill_str)])

skill_counts = Counter(skills_tokens)
top_skills = [skill for skill, _ in skill_counts.most_common(20)]  # You can increase to 50+ if needed

# Save top skills for external use (optional)
pd.Series(top_skills).to_csv("top_skills.csv", index=False)

# Compute skill_score for each employee
hot_skills_set = set(top_skills)

def compute_skill_score(skill_text):
    if pd.isna(skill_text):
        return 0.0
    person_skills = set([s.strip().title() for s in re.split(',|;', skill_text)])
    match_count = len(hot_skills_set & person_skills)
    return match_count / len(hot_skills_set) if hot_skills_set else 0.0

df['Skill_Score'] = df['Skills'].apply(compute_skill_score)


# Drop high-cardinality or non-numeric columns
X = df.drop(columns=['Resigned','EEID','Project ID','Project Name','Job Title','Field of Study','Gender','Business Unit','Department',
                           'Hire Date', 'Exit Date', 'Tech Stack','Skills'])
y = df['Resigned']

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train the LightGBM model
lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    is_unbalance=True,
    min_data_in_leaf=1,
    min_gain_to_split=0.0,
    random_state=42
)

lgb_model.fit(X_train_smote, y_train_smote)

# Save the trained model
with open("lightgbm_resignation_model.pkl", "wb") as f:
    pickle.dump(lgb_model, f)

# Save the feature names used in training (important for input matching in Streamlit)
with open("model_features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model and features saved successfully.")
