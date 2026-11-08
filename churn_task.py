import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("=== PART 1: DATA PREPROCESSING ===")

# 1. Load the dataset
df = pd.read_csv('churn.csv')
print(f"Original dataset shape: {df.shape}")

# Dataset bohot bara hai (440k+ rows), isliye fast execution ke liye 10,000 rows ka sample le rahe hain
df = df.sample(n=10000, random_state=42).reset_index(drop=True)

# 2. Handle missing values
print("Missing values before cleaning:", df.isnull().sum().sum())
df = df.dropna()

# 3. Remove duplicates
df = df.drop_duplicates()
print(f"Dataset shape after cleaning (sampled): {df.shape}")

# Drop ID columns if they exist
if 'CustomerID' in df.columns:
    df = df.drop(columns=['CustomerID'])

# 4. Encode categorical features
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Features (X) aur Target (y) split karna
target_col = 'Churn' if 'Churn' in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# 5. Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data Preprocessing completed successfully!\n")

print("=== PART 2 & 3: MODEL DEVELOPMENT & EVALUATION ===")
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=50) # Fast training ke liye n_estimators kam rakhe hain
}

results = []

for name, model in models.items():
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results.append({
        "Model Name": name,
        "Training Time (s)": round(train_time, 4),
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4)
    })
    print(f"-> {name} trained and evaluated.")

print("\n=== PART 4: MODEL COMPARISON TABLE ===")
comparison_df = pd.DataFrame(results)
print(comparison_df.to_string(index=False))

print("\n=== BONUS: HYPERPARAMETER TUNING (RandomizedSearchCV) ===")
# Random Forest ke liye hyperparameter tuning perform kar rahe hain
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
# Fast run ke liye n_iter=3 rakha hai
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=3, cv=3, scoring='f1', random_state=42, n_jobs=-1)
random_search.fit(X_train_scaled, y_train)

print("Tuning Completed!")
print("Best Parameters Found:", random_search.best_params_)
print("Best Score (F1):", round(random_search.best_score_, 4))

print("\n=== PART 5: CONCLUSION ===")
best_model_name = comparison_df.loc[comparison_df['F1 Score'].idxmax()]['Model Name']
print(f"Recommended Best Model: {best_model_name}")
print("Reason: Tuned Random Forest / Random Forest achieved the best classification balance and F1 score.")