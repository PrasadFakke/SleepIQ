# 🛌 SleepIQ - Sleep Health & Lifestyle Prediction System

A Machine Learning project that predicts sleep disorders based on health and lifestyle factors. The system uses **SMOTE, XGBoost, Stacking Ensemble, and SHAP explainability**, with an interactive **Streamlit web application**.

---


## 🌐 Streamlit Application

The Streamlit application provides an interactive interface where users can enter health and lifestyle information and receive a predicted sleep disorder category.

The application uses the saved model artifacts from the `artifacts/` folder.

### Live Demo

Try SleepIQ online: [sleepiq-project.streamlit.app](https://sleepiq-project.streamlit.app/)

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

* `images/shap_importance_bar.png`
* `images/shap_summary.png`
* `images/shap_waterfall.png`

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

---


## 👥 Team

- [Prasad Fakke](https://github.com/PrasadFakke)
- [Laukik Deshpande](https://github.com/laukik-26)

---

## ⚠️ Limitations

* **Small dataset** (~374 samples). Results can be optimistic; treat accuracy as indicative, not clinical-grade.
* **SMOTE** creates synthetic minority samples — helpful for training, but evaluation should always use the real (unbalanced) test set.
* **No external validation**. Performance is reported on a single stratified hold-out split.
* **Rule-based recommendations and guided Q&A** — not an LLM or medical decision system.
* Heart rate and very low daily steps are capped during preprocessing to reduce outlier influence; this is a pragmatic choice, not a clinical rule.

## ⚠️ Disclaimer

This project is developed for **academic and educational purposes**. The predictions should not be considered a substitute for professional medical diagnosis or treatment.

---

<div align="center">

### 🛌 SleepIQ - Sleep Health & Lifestyle Prediction System

**Built with Python, Machine Learning, SHAP & Streamlit ❤️**

⭐ Star the repository if you find it useful!

</div>


