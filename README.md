# Gallstone Disease Prediction Using Machine Learning

## 📌 Project Overview

This project develops a machine learning–based prediction system for detecting gallstone disease using clinical data. The study benchmarks multiple supervised learning models to identify the most accurate classifier for predicting gallstone risk based on patient medical records.

The model aims to support early diagnosis and decision-making in healthcare by providing data-driven predictions from clinical features.

---

## 🎯 Objectives

* Develop predictive models for gallstone disease detection.
* Compare multiple supervised machine learning algorithms.
* Identify the best-performing model based on evaluation metrics.
* Provide a reproducible workflow for clinical data prediction.

---

## 📊 Dataset

The dataset contains clinical records used to predict gallstone occurrence.

### Features include:

* Patient clinical measurements
* Biochemical indicators
* Health-related attributes

**Dataset file:** `gallstone_selected.csv`

---

## ⚙️ Machine Learning Workflow

### 1. Data Preprocessing

* Data loading and cleaning
* Feature selection
* Handling missing values
* Data normalization/scaling

### 2. Model Training

Multiple supervised learning models were implemented and compared:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine
* K-Nearest Neighbors
* AdaBoost
* Other classifiers (as implemented in the notebook)

### 3. Model Evaluation

Models were evaluated using:

* Accuracy score
* Confusion matrix
* Classification metrics
* Performance comparison

### 4. Model Selection

The best-performing model was selected based on predictive performance.

---

## 🧠 Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* Matplotlib / Seaborn
* Jupyter Notebook

---

## 📂 Project Structure

```
├── Gallstone model prediction.ipynb   # Main notebook
├── gallstone_selected.csv             # Dataset
├── README.md                          # Project documentation
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```
git clone https://github.com/yourusername/gallstone-prediction.git
cd gallstone-prediction
```

### 2. Install Dependencies

```
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Run the Notebook

```
jupyter notebook
```

Open:

```
Gallstone model prediction.ipynb
```

---

## 📈 Results

The project evaluates multiple machine learning classifiers and identifies the most accurate model for gallstone disease prediction based on clinical data.

Key findings include:

* Performance comparison of models
* Identification of optimal classifier
* Data-driven insights for medical prediction

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

This project is for academic and research purposes.
