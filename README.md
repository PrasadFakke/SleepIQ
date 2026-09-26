# 🛌 SleepIQ - Sleep Health & Lifestyle Prediction System

A Machine Learning project that predicts sleep disorder categories based on health and lifestyle factors. The system uses **SMOTE, XGBoost, Stacking Ensemble, and SHAP explainability** with an interactive **Streamlit web application**.

---

## 🌐 Streamlit Application

The Streamlit application provides an interactive interface where users can enter health and lifestyle information and receive a predicted sleep disorder category.

The application uses the saved model artifacts from the `artifacts/` folder.

### Live URL

Try SleepIQ online: [sleepiq-project.streamlit.app](https://sleepiq-project.streamlit.app/)

### Project Screenshot

<img width="1916" height="827" alt="SleepIQ Streamlit Application" src="https://github.com/user-attachments/assets/5612ea4f-3446-4220-bfca-82f11702a47f" />

---

## ✨ Features

* 🤖 Sleep disorder prediction
* ⚖️ Class balancing using **SMOTE**
* 🧩 **Stacking Ensemble** for multi-model classification
* 🚀 **XGBoost** model
* 🔍 **SHAP** for model explainability
* 📊 Model evaluation and visualizations
* 🌐 Interactive **Streamlit web application**

---

## 🛠️ Technologies

* **Python**
* **Pandas & NumPy**
* **Scikit-learn**
* **XGBoost**
* **Imbalanced-learn (SMOTE)**
* **SHAP**
* **Matplotlib & Seaborn**
* **Streamlit**

---

## 📊 Dataset

The project uses the **Sleep Health and Lifestyle Dataset**, containing health and lifestyle information such as:

* Age and Gender
* Sleep Duration and Quality
* Physical Activity
* Stress Level
* BMI
* Blood Pressure
* Heart Rate
* Daily Steps

### Target Classes

`Healthy` • `Insomnia` • `Sleep Apnea`

**Source:** [Sleep Health and Lifestyle Dataset - Kaggle](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)

---

## 🧠 Machine Learning Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Feature Preparation
   ↓
Train-Test Split
   ↓
SMOTE
   ↓
XGBoost + Stacking Ensemble
   ↓
Model Evaluation
   ↓
SHAP Explainability
   ↓
Streamlit Web Application
```

---

## 🔄 Prediction Flow

1. User enters health and lifestyle information.
2. Input features are preprocessed using the saved scaler and feature configuration.
3. The trained Stacking Ensemble generates the predicted sleep disorder category.
4. SHAP provides feature-level explanations for the prediction.
5. The Streamlit interface displays the prediction and supporting information.

---

## 📁 Project Structure

```text
SleepIQ/
│
├── artifacts/          # saved models & preprocessors
│   ├── feature_cols.pkl
│   ├── le_target.pkl
│   ├── scaler.pkl
│   ├── shap_explainer.pkl
│   ├── stacking_model.pkl
│   └── xgb_model.pkl
│
├── images/             # evaluation & SHAP plots
│   ├── confusion_matrix_stacking.png
│   ├── shap_importance_bar.png
│   ├── shap_summary.png
│   ├── shap_waterfall.png
│   └── smote_distribution.png
│
├── app.py
├── train_model.py
├── utils.py
├── requirements.txt
└── Sleep_health_and_lifestyle_dataset.csv
```

### 📦 Model Artifacts

| File | Purpose |
|---|---|
| `feature_cols.pkl` | Stores the features used by the model |
| `le_target.pkl` | Target-label encoding |
| `scaler.pkl` | Feature scaling |
| `shap_explainer.pkl` | Saved SHAP explainer |
| `stacking_model.pkl` | Trained Stacking Ensemble |
| `xgb_model.pkl` | Trained XGBoost model |

---

## 📈 Model Evaluation

The project evaluates the trained models using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Additional visualizations include:

* `images/confusion_matrix_stacking.png`
* `images/smote_distribution.png`

### Results

The Stacking Ensemble achieved **93%+ accuracy** on the stratified hold-out test set.


---

## 🔍 SHAP Explainability

The project uses **SHAP (SHapley Additive exPlanations)** to understand how different features contribute to model predictions.

The repository includes:

* `images/shap_importance_bar.png`
* `images/shap_summary.png`
* `images/shap_waterfall.png`

These visualizations help interpret the contribution of individual features toward model predictions.

---

## ⚙️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

This trains the required models and generates the model artifacts used by the application.

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit, usually:

```text
http://localhost:8501
```

---

## 👥 Team

* [Prasad Fakke](https://github.com/PrasadFakke)
* [Laukik Deshpande](https://github.com/laukik-26)

---

## ⚠️ Limitations

* **Small dataset** (~374 samples). Results can be optimistic; treat accuracy as indicative, not clinical-grade.
* **SMOTE** creates synthetic minority samples. It helps training, but evaluation should always use the real, unbalanced test set.
* **No external validation.** Performance is reported on a single stratified hold-out split.
* **Rule-based recommendations and guided Q&A** are not an LLM or medical decision system.
* Heart rate and very low daily-step values are capped during preprocessing to reduce the influence of extreme values. These thresholds are project-specific preprocessing choices and are not clinical guidelines.

---

## ⚠️ Disclaimer

This project is developed for **academic and educational purposes**. The predictions should not be considered a substitute for professional medical diagnosis or treatment.

---

<div align="center">

<b>🛌 SleepIQ - Sleep Health Dashboard</b><br>
<b>Built with Python, Machine Learning, SHAP, and Streamlit.</b><br>
⭐ <b>Star the repository if you find it useful!</b>

</div>
