# 🛌 Sleep Health & Lifestyle Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge\&logo=xgboost\&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-8E44AD?style=for-the-badge)

**A Machine Learning system for predicting sleep disorders from health and lifestyle factors using SMOTE, ensemble learning and SHAP-based explainability.**

</div>

---

## 📌 About the Project

Sleep health is influenced by several factors such as sleep duration, quality of sleep, stress level, physical activity, BMI, blood pressure and heart rate.

This project develops a **Machine Learning based Sleep Disorder Prediction System** that analyzes these health and lifestyle parameters and predicts the possible sleep disorder category.

The system combines **data preprocessing, class balancing using SMOTE, machine learning classification, ensemble learning and model explainability using SHAP**.

A **Streamlit web application** provides an interactive interface where users can enter their details and obtain a prediction.

> **Note:** This project is intended for academic and educational purposes and should not be considered a medical diagnostic system.

---

## 🎯 Objectives

* Predict sleep disorder categories using Machine Learning.
* Analyze the relationship between lifestyle factors and sleep health.
* Handle class imbalance using **SMOTE**.
* Compare multiple classification models.
* Improve prediction using a **Stacking Ensemble**.
* Explain model predictions using **SHAP**.
* Provide an easy-to-use web interface using **Streamlit**.

---

## ✨ Key Features

| Feature                | Description                                                    |
| ---------------------- | -------------------------------------------------------------- |
| 🤖 Machine Learning    | Classification models for sleep disorder prediction            |
| ⚖️ SMOTE               | Handles class imbalance in the training data                   |
| 🧩 Stacking Ensemble   | Combines multiple ML models for improved prediction            |
| 🔍 SHAP Explainability | Shows which features influence predictions                     |
| 🌐 Streamlit App       | Interactive interface for making predictions                   |
| 📊 Data Visualization  | Includes model and data analysis visualizations                |
| 💡 Health Insights     | Provides interpretable information based on prediction factors |

---

## 🧠 Machine Learning Pipeline

```text
                    Sleep Health Dataset
                            │
                            ▼
                   Data Preprocessing
                            │
                            ▼
                    Feature Engineering
                            │
                            ▼
                    Train-Test Split
                            │
                            ▼
                         SMOTE
                            │
                            ▼
                 Multiple ML Classifiers
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
        Random Forest     SVM          XGBoost
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                   Stacking Ensemble
                            │
                            ▼
                    Model Evaluation
                            │
                            ▼
                    SHAP Explainability
                            │
                            ▼
                    Streamlit Web App
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* XGBoost
* Imbalanced-learn
* SMOTE

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Explainable AI

* SHAP

### Web Application

* Streamlit

---

## 📊 Dataset

The project uses the **Sleep Health and Lifestyle Dataset**.

The dataset contains information related to:

* Age
* Gender
* Occupation
* Sleep Duration
* Quality of Sleep
* Physical Activity Level
* Stress Level
* BMI Category
* Blood Pressure
* Heart Rate
* Daily Steps
* Sleep Disorder

### Target Classes

The prediction target contains three categories:

```text
Healthy
Insomnia
Sleep Apnea
```

The dataset contains **374 records**.

### Dataset Source

[Kaggle - Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)

---

## ⚖️ Handling Class Imbalance with SMOTE

The dataset contains an unequal number of samples across the sleep disorder categories.

To reduce the effect of class imbalance during model training, **Synthetic Minority Oversampling Technique (SMOTE)** is applied to the training data.

This generates synthetic samples for minority classes and helps the models learn from a more balanced training distribution.

---

## 🤖 Models

The project evaluates multiple classification approaches, including:

* Logistic Regression
* Random Forest
* Support Vector Machine
* Gradient Boosting
* XGBoost
* Stacking Ensemble

The **Stacking Ensemble** combines predictions from multiple base learners and uses a meta-classifier to produce the final prediction.

---

## 🔍 Explainable AI with SHAP

Machine Learning predictions can be difficult to interpret.

To make the model more understandable, this project uses **SHAP (SHapley Additive exPlanations)**.

SHAP helps identify how individual features contribute to a model prediction.

The repository contains generated SHAP visualizations such as:

* `shap_importance_bar.png`
* `shap_summary.png`
* `shap_waterfall.png`

These visualizations help understand which health and lifestyle factors have the greatest influence on model predictions.

---

## 📈 Model Evaluation

The project evaluates the trained models using standard classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

A confusion matrix visualization is also included in the repository:

`confusion_matrix_stacking.png`

Additional analysis of the effect of SMOTE is represented through:

`smote_distribution.png`

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit application.

Users can provide relevant health and lifestyle information through the interface, after which the trained model generates a predicted sleep disorder category.

The application is designed to make the Machine Learning model accessible without requiring users to run individual Python scripts.

### Run the application

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## 📁 Project Structure

```text
SleepIQ/
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── Sleep_health_and_lifestyle_dataset.csv
│
├── artifacts/
│   └── trained model files
│
├── confusion_matrix_stacking.png
├── shap_importance_bar.png
├── shap_summary.png
├── shap_waterfall.png
├── smote_distribution.png
│
└── README.md
```

### File Description

| File / Folder                            | Purpose                                          |
| ---------------------------------------- | ------------------------------------------------ |
| `app.py`                                 | Streamlit web application                        |
| `train_model.py`                         | Model training and preprocessing                 |
| `requirements.txt`                       | Python dependencies                              |
| `artifacts/`                             | Saved trained models and preprocessing artifacts |
| `Sleep_health_and_lifestyle_dataset.csv` | Dataset                                          |
| `confusion_matrix_stacking.png`          | Stacking model confusion matrix                  |
| `shap_importance_bar.png`                | SHAP feature importance                          |
| `shap_summary.png`                       | SHAP summary visualization                       |
| `shap_waterfall.png`                     | SHAP waterfall visualization                     |
| `smote_distribution.png`                 | Class distribution visualization                 |

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

---

## 🧪 Training the Model

If you want to retrain the models from the dataset:

```bash
python train_model.py
```

The training process performs the required preprocessing and model training and stores the generated model artifacts in the `artifacts/` directory.

After training, the Streamlit application can be launched using:

```bash
streamlit run app.py
```

---

## 📊 Project Results

The project generates several evaluation and explainability outputs.

### Confusion Matrix

The stacking ensemble's classification performance can be inspected using:

```text
confusion_matrix_stacking.png
```

### SMOTE Distribution

The effect of class balancing can be visualized using:

```text
smote_distribution.png
```

### SHAP Analysis

Feature contributions can be explored through:

```text
shap_importance_bar.png
shap_summary.png
shap_waterfall.png
```

---

## 🔮 Future Enhancements

* Integration with wearable devices and fitness trackers.
* Larger and more diverse sleep-health datasets.
* Time-series analysis of sleep patterns.
* Cloud deployment of the Streamlit application.
* REST API integration using FastAPI.
* Additional explainability techniques such as LIME.
* Improved personalization of sleep recommendations.

---

## ⚠️ Disclaimer

This project is developed for **educational and academic purposes**.

The predictions generated by this system should **not be used as a substitute for professional medical diagnosis or treatment**. Users with concerns about their sleep health should consult a qualified healthcare professional.

---

## 👥 Team

**Project:** Sleep Health & Lifestyle Prediction System

**Institution:** Bharatiya Vidya Bhavan's Sardar Patel Institute of Technology (SPIT), Mumbai

**Subject:** Pattern Discovery and Statistics

---

## 🙏 Acknowledgements

* [Kaggle Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
* Scikit-learn
* XGBoost
* SHAP
* Imbalanced-learn
* Streamlit
* Python open-source community

---

<div align="center">

### 🛌 Sleep Health & Lifestyle Prediction System

**Built with Python, Machine Learning, SHAP & Streamlit ❤️**

⭐ Star the repository if you find the project useful!

</div>
