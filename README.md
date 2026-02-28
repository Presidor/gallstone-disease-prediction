# Gallstone Disease Prediction Using Machine Learning

## 📌 Project Overview

This project develops a **machine learning–based prediction system** for detecting gallstone disease using clinical, laboratory, and bioimpedance data. The study benchmarks multiple supervised learning models to identify the most accurate classifier for predicting gallstone risk based on patient medical records.

The model aims to support **early diagnosis and decision-making in healthcare** by providing data-driven predictions from clinical features.

---

## 🎯 Objectives

* Build predictive models for gallstone disease detection.  
* Compare multiple supervised machine learning algorithms.  
* Identify the best-performing model based on evaluation metrics.  
* Provide a reproducible workflow for clinical data prediction.  

---

## 📊 Dataset

The dataset contains **319 patient records** with 12 features, used to predict gallstone occurrence.

### Features include:
* **Bioimpedance**: Total Body Fat Ratio (TBFR), Bone Mass (BM), Extracellular/Intracellular Water.  
* **Laboratory**: Vitamin D, ALT, AST, CRP, Hemoglobin.  
* **Comorbidities**: Hyperlipidemia.  
* **Demographics**: Gender.  
* **Target Variable**: Gallstone Status (0 = No Gallstone, 1 = Gallstone).  

**Dataset file:** `gallstone_selected.csv`

---

## ⚙️ Machine Learning Workflow

### 1. Data Preprocessing
* Data loading and cleaning  
* Feature selection  
* Handling missing values (none in this dataset)  
* Normalization/scaling  

### 2. Model Training
Multiple supervised learning models were implemented and compared:
* Logistic Regression  
* Decision Tree  
* Random Forest  
* Support Vector Machine (SVM)  
* Neural Networks (MLP)  
* AdaBoost  
* XGBoost  
* Gradient Boosting  

### 3. Model Evaluation
Models were evaluated using:
* Accuracy score  
* Confusion matrix  
* Classification metrics (precision, recall, F1-score)  
* ROC-AUC curves  

### 4. Model Selection
The best-performing model was selected based on predictive performance.  
- **Logistic Regression** achieved ~90% accuracy and ROC-AUC ≈ 0.92.  
- **AdaBoost** achieved ~87.5% accuracy and ROC-AUC ≈ 0.90.  

---

## 🧠 Technologies Used
* Python  
* Scikit-learn  
* Pandas  
* NumPy  
* Matplotlib / Seaborn / Plotly  
* XGBoost  
* SHAP (for feature importance)

---

## 📂 Project Structure
```
├── Gallstone model prediction.pdf     # Full analysis & results
├── gallstone_selected.csv             # Dataset
├── Gallstone model prediction.ipynb   # Main notebook
├── app.py                             # Streamlit app (optional)
├── requirements.txt                   # Dependencies
└── README.md                          # Project documentation
```

---

## 📈 Results

The project evaluates multiple machine learning classifiers and identifies the most accurate model for gallstone disease prediction based on clinical data.

**Key findings:**
* Logistic Regression achieved the highest accuracy (~90%).  
* CRP, TBFR, and Vitamin D were the most influential predictors.  
* SHAP analysis provided interpretability of feature importance.  

---

## 🔬 Research Contribution

This work demonstrates how machine learning can assist medical diagnosis by:
* Improving disease prediction accuracy  
* Supporting clinical decision-making  
* Providing scalable predictive healthcare solutions  

---

## 👤 Author

**Chinonso Athanasius**  
* BSc Geological Science — Nnamdi Azikiwe University  
* Data Science & Applied AI Practitioner  
* Machine Learning Researcher  

---

## 📜 License

This project is for **academic and research purposes only**.  
It is **not a medical diagnostic tool**. Always consult healthcare professionals for medical advice. 
