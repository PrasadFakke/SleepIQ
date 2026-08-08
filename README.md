# 🛌 Sleep Health & Lifestyle Prediction System

A Machine Learning project that predicts sleep disorders based on health and lifestyle factors. The system uses **SMOTE, multiple classification models, Stacking Ensemble, and SHAP explainability**, with an interactive **Streamlit web application**.

> **Academic project for educational purposes. This system is not a medical diagnostic tool.**

## ✨ Features

* 🤖 Sleep disorder prediction
* ⚖️ Class balancing using **SMOTE**
* 🧩 **Stacking Ensemble** for classification
* 🔍 **SHAP** for model explainability
* 📊 Model evaluation and visualizations
* 🌐 Interactive **Streamlit** web application

## 🛠️ Technologies

* **Python**
* **Pandas & NumPy**
* **Scikit-learn**
* **XGBoost**
* **Imbalanced-learn (SMOTE)**
* **SHAP**
* **Matplotlib & Seaborn**
* **Streamlit**

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

## 📁 Project Structure

```text
Sleep-Health-Prediction/
│
├── app.py
├── artifacts/
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

## ⚙️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## 🔍 Explainability

The project uses **SHAP (SHapley Additive exPlanations)** to understand how different features influence model predictions.

Generated visualizations include:

* `shap_importance_bar.png`
* `shap_summary.png`
* `shap_waterfall.png`

## 📈 Results

The project evaluates the trained models using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The repository also includes the stacking ensemble confusion matrix and SMOTE distribution visualization.


## 👥 Team

- [Prasad Fakke](https://github.com/PrasadFakke)
- [Laukik Deshpande](https://github.com/laukik-26)

---

⭐ **If you find this project useful, consider starring the repository.**
