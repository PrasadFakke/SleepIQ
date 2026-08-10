# 🛌 SleepIQ - Sleep Health & Lifestyle Prediction System

A Machine Learning project that predicts sleep disorders based on health and lifestyle factors. The system uses **SMOTE, XGBoost, Stacking Ensemble, and SHAP explainability**, with an interactive **Streamlit web application**.

---

## ✨ Features

* 🤖 Sleep disorder prediction
* ⚖️ Class balancing using **SMOTE**
* 🧩 **Stacking Ensemble** for improved classification
* 🚀 **XGBoost** model
* 🔍 **SHAP** for model explainability
* 📊 Model evaluation and visualizations
* 🌐 Interactive **Streamlit** web application

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

**Dataset:** [Kaggle - Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)

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

## 📁 Project Structure

```text
SleepIQ/
│
│
├── artifacts/
│   ├── feature_cols.pkl
│   ├── le_target.pkl
│   ├── scaler.pkl
│   ├── shap_explainer.pkl
│   ├── stacking_model.pkl
│   └── xgb_model.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
├── Sleep_health_and_lifestyle_dataset.csv
│
├── confusion_matrix_stacking.png
├── shap_importance_bar.png
├── shap_summary.png
├── shap_waterfall.png
└── smote_distribution.png
```

### 📦 Model Artifacts

| File                 | Purpose                               |
| -------------------- | ------------------------------------- |
| `feature_cols.pkl`   | Stores the features used by the model |
| `le_target.pkl`      | Target-label encoding                 |
| `scaler.pkl`         | Feature scaling                       |
| `shap_explainer.pkl` | Saved SHAP explainer                  |
| `stacking_model.pkl` | Trained Stacking Ensemble             |
| `xgb_model.pkl`      | Trained XGBoost model                 |

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

## 🔍 SHAP Explainability

The project uses **SHAP (SHapley Additive exPlanations)** to understand how different features contribute to model predictions.

The repository includes:

* `shap_importance_bar.png`
* `shap_summary.png`
* `shap_waterfall.png`

---

## 📈 Model Evaluation

The project evaluates the trained models using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Additional visualizations include:

* `confusion_matrix_stacking.png`
* `smote_distribution.png`

---

## 🌐 Streamlit Application

The Streamlit application provides an interactive interface where users can enter health and lifestyle information and receive a predicted sleep disorder category.

The application uses the saved model artifacts from the `artifacts/` folder.

---

## 👥 Team

- [Prasad Fakke](https://github.com/PrasadFakke)
- [Laukik Deshpande](https://github.com/laukik-26)

---

## ⚠️ Disclaimer

This project is developed for **academic and educational purposes**. The predictions should not be considered a substitute for professional medical diagnosis or treatment.

---

<div align="center">

### 🛌 SleepIQ - Sleep Health & Lifestyle Prediction System

**Built with Python, Machine Learning, SHAP & Streamlit ❤️**

⭐ Star the repository if you find it useful!

</div>

