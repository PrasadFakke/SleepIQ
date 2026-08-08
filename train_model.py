"""
=============================================================
  Sleep Health & Lifestyle ML Project
  train_model.py  —  Run this FIRST to train & save models
  Usage:  python train_model.py
=============================================================
"""
import matplotlib
matplotlib.use('Agg')
# ──────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # Non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               StackingClassifier)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

print("✅ All libraries imported successfully!")

# ──────────────────────────────────────────────
# STEP 1 — Load & Preprocess Dataset
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1 — Loading & Preprocessing Data")
print("="*60)

data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
data.drop('Person ID', axis=1, inplace=True)

# Group infrequent occupations
infrequent = ['Manager', 'Sales Representative', 'Scientist', 'Software Engineer']
data['Occupation'] = data['Occupation'].replace(infrequent, 'Others')

# Fix Quality of Sleep (merge low-count bin)
data['Quality of Sleep'] = data['Quality of Sleep'].replace({4: 4.5, 5: 4.5})

# Normalize BMI labels
data['BMI Category'] = data['BMI Category'].replace('Normal Weight', 'Normal')
bmi_mapping = {'Normal': 0, 'Overweight': 1, 'Obese': 2}
data['BMI Category'] = data['BMI Category'].map(bmi_mapping)

# Split Blood Pressure into two columns
data[['Systolic Pressure', 'Diastolic Pressure']] = (
    data['Blood Pressure'].str.split('/', expand=True).astype(int)
)
data.drop('Blood Pressure', axis=1, inplace=True)

# Cap Heart Rate outliers
data['Heart Rate'] = data['Heart Rate'].replace(
    [78, 80, 81, 82, 83, 84, 85, 86], 78
)

# Cap very low Daily Steps
data.loc[data['Daily Steps'] < 5000, 'Daily Steps'] = 4800

# Fill missing Sleep Disorder → Healthy
data['Sleep Disorder'] = data['Sleep Disorder'].fillna('Healthy')

# ── Feature Engineering ──
data['Sleep Efficiency']  = data['Sleep Duration'] / data['Quality of Sleep']
data['Health Risk Index'] = (
    data['Stress Level'] * data['BMI Category'] + data['Heart Rate'] / 10
)

# ── Encode categoricals ──
le = LabelEncoder()
data['Gender']     = le.fit_transform(data['Gender'])       # Male=1, Female=0
data['Occupation'] = le.fit_transform(data['Occupation'])

le_target = LabelEncoder()
data['Sleep Disorder Encoded'] = le_target.fit_transform(data['Sleep Disorder'])

print("Classes:", le_target.classes_)   # [Healthy, Insomnia, Sleep Apnea]
print("Dataset shape:", data.shape)

# ──────────────────────────────────────────────
# STEP 2 — Define Features & Split
# ──────────────────────────────────────────────
feature_cols = [
    'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
    'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate',
    'Daily Steps', 'Systolic Pressure', 'Diastolic Pressure',
    'Sleep Efficiency', 'Health Risk Index'
]

X = data[feature_cols]
y = data['Sleep Disorder Encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTrain: {X_train_scaled.shape[0]} samples | Test: {X_test_scaled.shape[0]} samples")

# ──────────────────────────────────────────────
# STEP 3 — SMOTE (Handle Class Imbalance)
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3 — SMOTE: Handling Class Imbalance")
print("="*60)

before = pd.Series(y_train).value_counts().sort_index()
print("\n📊 Before SMOTE:")
for idx, count in before.items():
    print(f"  {le_target.classes_[idx]:12s}: {count} samples")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

after = pd.Series(y_train_smote).value_counts().sort_index()
print("\n📊 After SMOTE:")
for idx, count in after.items():
    print(f"  {le_target.classes_[idx]:12s}: {count} samples")

# Save SMOTE comparison chart
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = ['#2ecc71', '#e74c3c', '#3498db']

axes[0].bar(le_target.classes_, before.values, color=colors, edgecolor='black')
axes[0].set_title('Before SMOTE', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(before.values):
    axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')

axes[1].bar(le_target.classes_, after.values, color=colors, edgecolor='black')
axes[1].set_title('After SMOTE', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Count')
for i, v in enumerate(after.values):
    axes[1].text(i, v + 1, str(v), ha='center', fontweight='bold')

plt.suptitle('Class Distribution Before vs After SMOTE', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('smote_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ smote_distribution.png saved!")

# Quick SMOTE comparison table
print("\n📋 Model Accuracy Comparison (No SMOTE vs With SMOTE):")
print(f"{'Model':<22} | {'No SMOTE':>10} | {'With SMOTE':>10}")
print("-" * 50)
for name, mdl in {
    'Random Forest'    : RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Reg.'    : LogisticRegression(max_iter=1000, random_state=42),
}.items():
    mdl.fit(X_train_scaled, y_train)
    acc_before = accuracy_score(y_test, mdl.predict(X_test_scaled))
    mdl.fit(X_train_smote, y_train_smote)
    acc_after  = accuracy_score(y_test, mdl.predict(X_test_scaled))
    diff = acc_after - acc_before
    sign = "▲" if diff > 0 else ("▼" if diff < 0 else "─")
    print(f"{name:<22} | {acc_before:>10.4f} | {acc_after:>10.4f}  {sign}{abs(diff):.4f}")

# ──────────────────────────────────────────────
# STEP 4 — XGBoost + Stacking Ensemble
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4 — XGBoost & Stacking Ensemble")
print("="*60)

# ---- XGBoost ----
print("\n🚀 Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators     = 200,
    max_depth        = 5,
    learning_rate    = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    eval_metric      = 'mlogloss',
    random_state     = 42
)
xgb_model.fit(X_train_smote, y_train_smote)
xgb_preds = xgb_model.predict(X_test_scaled)
xgb_acc   = accuracy_score(y_test, xgb_preds)
cv_xgb    = cross_val_score(xgb_model, X_train_smote, y_train_smote, cv=5, scoring='accuracy')
print(f"  XGBoost Test Acc : {xgb_acc:.4f}")
print(f"  XGBoost CV Mean  : {cv_xgb.mean():.4f} ± {cv_xgb.std():.4f}")

# ---- Stacking Ensemble ----
print("\n🔗 Building Stacking Ensemble (RF + SVM + GB + XGB → LogReg)...")
base_estimators = [
    ('rf',  RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('gb',  GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        eval_metric='mlogloss', random_state=42
    )),
]

stacking_model = StackingClassifier(
    estimators      = base_estimators,
    final_estimator = LogisticRegression(max_iter=1000, random_state=42),
    cv              = 5,
    stack_method    = 'predict_proba',
    n_jobs          = -1
)

stacking_model.fit(X_train_smote, y_train_smote)
stack_preds = stacking_model.predict(X_test_scaled)
stack_acc   = accuracy_score(y_test, stack_preds)
cv_stack    = cross_val_score(stacking_model, X_train_smote, y_train_smote, cv=5, scoring='accuracy')
print(f"  Stacking Test Acc : {stack_acc:.4f}")
print(f"  Stacking CV Mean  : {cv_stack.mean():.4f} ± {cv_stack.std():.4f}")

# Comparison table
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rf_base.fit(X_train_scaled, y_train)
rf_acc = accuracy_score(y_test, rf_base.predict(X_test_scaled))
cv_rf  = cross_val_score(rf_base, X_train_scaled, y_train, cv=5, scoring='accuracy')

print(f"\n{'Model':<28} | {'Test Acc':>9} | {'CV Mean':>9} | {'CV Std':>8}")
print("-" * 62)
for name, (acc, cv_m, cv_s) in {
    'Random Forest (baseline)':    (rf_acc,    cv_rf.mean(),    cv_rf.std()),
    'XGBoost (+ SMOTE)':           (xgb_acc,   cv_xgb.mean(),   cv_xgb.std()),
    'Stacking Ensemble (+ SMOTE)': (stack_acc, cv_stack.mean(), cv_stack.std()),
}.items():
    print(f"{name:<28} | {acc:>9.4f} | {cv_m:>9.4f} | {cv_s:>8.4f}")

print(f"\n📋 Stacking Ensemble — Detailed Report:")
print(classification_report(y_test, stack_preds, target_names=le_target.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, stack_preds)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_)
plt.title('Confusion Matrix — Stacking Ensemble', fontsize=13)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_stacking.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ confusion_matrix_stacking.png saved!")

# ──────────────────────────────────────────────
# STEP 5 — SHAP Explainability
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5 — SHAP Explainability (using XGBoost)")
print("="*60)

print("\n🔍 Computing SHAP values ...")
explainer   = shap.TreeExplainer(xgb_model)
X_test_df   = pd.DataFrame(X_test_scaled, columns=feature_cols)
shap_values = explainer.shap_values(X_test_df)

# Summary plot
plt.figure()
shap.summary_plot(shap_values, X_test_df, class_names=le_target.classes_, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ shap_summary.png saved!")

# Bar plot
plt.figure()
shap.summary_plot(shap_values, X_test_df, plot_type='bar',
                  class_names=le_target.classes_, show=False)
plt.tight_layout()
plt.savefig('shap_importance_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ shap_importance_bar.png saved!")

# Waterfall — single sample
sample_idx  = 0
pred_class  = int(xgb_model.predict(X_test_df.iloc[[sample_idx]])[0])
pred_label  = le_target.classes_[pred_class]

explanation = shap.Explanation(
    values        = shap_values[pred_class][sample_idx],
    base_values   = explainer.expected_value[pred_class],
    data          = X_test_df.iloc[sample_idx].values,
    feature_names = feature_cols
)

plt.figure()
shap.plots.waterfall(explanation, show=False)
plt.title(f"SHAP Waterfall — Sample 0 | Predicted: {pred_label}", fontsize=11)
plt.tight_layout()
plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ shap_waterfall.png saved! (Prediction: {pred_label})")

# ──────────────────────────────────────────────
# STEP 6 — Save All Artifacts
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6 — Saving Model Artifacts")
print("="*60)

os.makedirs('artifacts', exist_ok=True)

joblib.dump(stacking_model, 'artifacts/stacking_model.pkl')
joblib.dump(xgb_model,      'artifacts/xgb_model.pkl')
joblib.dump(scaler,         'artifacts/scaler.pkl')
joblib.dump(le_target,      'artifacts/le_target.pkl')
joblib.dump(feature_cols,   'artifacts/feature_cols.pkl')
joblib.dump(explainer,      'artifacts/shap_explainer.pkl')

print("  ✅ stacking_model.pkl")
print("  ✅ xgb_model.pkl")
print("  ✅ scaler.pkl")
print("  ✅ le_target.pkl")
print("  ✅ feature_cols.pkl")
print("  ✅ shap_explainer.pkl")
print("\n🎉 Training complete! All models and charts saved.")


# ══════════════════════════════════════════════
# UTILITY FUNCTIONS  (imported by app.py)
# ══════════════════════════════════════════════

def get_recommendations(stress, sleep_duration, bmi_category, physical_activity, disorder):
    """
    Returns (disorder_name: str, recommendations: list[str]).
    disorder: int  0=Healthy | 1=Insomnia | 2=Sleep Apnea
    """
    class_names   = {0: 'Healthy', 1: 'Insomnia', 2: 'Sleep Apnea'}
    disorder_name = class_names.get(int(disorder), 'Unknown')
    recs          = []

    # Stress
    if stress >= 7:
        recs += [
            "😰 HIGH STRESS: Practice deep breathing or meditation daily.",
            "📵 Limit screen time 1 hour before bed.",
            "📓 Try journaling to offload thoughts before sleep.",
        ]
    elif stress >= 5:
        recs += [
            "😐 MODERATE STRESS: Light yoga or stretching before bed.",
            "🕐 Maintain a consistent sleep schedule.",
        ]
    else:
        recs.append("✅ Good stress management — keep it up!")

    # Sleep duration
    if sleep_duration < 6:
        recs += [
            "😴 INSUFFICIENT SLEEP: Aim for 7–9 hours per night.",
            "☕ Avoid caffeine after 3 PM.",
        ]
    elif sleep_duration > 9:
        recs.append("💤 EXCESSIVE SLEEP may signal a disorder — consult a specialist.")
    else:
        recs.append("✅ Sleep duration is in the healthy range (7–9 hrs).")

    # BMI
    if bmi_category == 2:
        recs += [
            "⚖️ OBESE BMI: Weight management may reduce Sleep Apnea risk.",
            "🥗 Consult a nutritionist for a personalised plan.",
        ]
    elif bmi_category == 1:
        recs.append("⚖️ OVERWEIGHT: Regular cardio exercise is recommended.")
    else:
        recs.append("✅ BMI is in the normal range.")

    # Physical activity
    if physical_activity < 30:
        recs += [
            "🏃 LOW ACTIVITY: Aim for 30 min of moderate exercise daily.",
            "🌙 Avoid intense workouts within 2 hours of bedtime.",
        ]
    elif physical_activity >= 60:
        recs.append("✅ Great physical activity level — well done!")
    else:
        recs.append("🏃 Moderate activity — try to build up gradually.")

    # Disorder-specific
    if disorder == 1:
        recs += [
            "🌙 INSOMNIA: Consider Cognitive Behavioural Therapy (CBT-I).",
            "❄️ Keep bedroom cool, dark, and quiet.",
            "🚫 Avoid naps longer than 20 minutes.",
        ]
    elif disorder == 2:
        recs += [
            "😤 SLEEP APNEA: Consult a doctor about CPAP therapy.",
            "🛌 Sleep on your side instead of your back.",
            "🚫 Avoid alcohol and sedatives before sleep.",
        ]
    else:
        recs.append("🎉 HEALTHY: Maintain your current lifestyle habits!")

    return disorder_name, recs


def sleep_risk_score(stress, sleep_duration, bmi_category, physical_activity):
    """
    Calculate a 0–100 sleep risk score.
    Returns: (score: float, risk_level: str)
    """
    score  = 0
    score += min(stress * 3.75, 30)
    score += 25 if sleep_duration < 6 else (15 if sleep_duration < 7 else 0)
    score += bmi_category * 12.5
    score += max(0, (60 - physical_activity) / 60 * 20)
    score  = min(score, 100)

    if score < 30:
        risk_level = "🟢 Low Risk"
    elif score < 60:
        risk_level = "🟡 Moderate Risk"
    else:
        risk_level = "🔴 High Risk"

    return round(score, 1), risk_level
