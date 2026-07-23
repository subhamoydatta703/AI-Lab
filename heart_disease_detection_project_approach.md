# Heart Disease Risk Classifier

*Project Approach and Plan*

## 1. What We Are Going to Build

- Predicts a patient's risk of heart disease using values taken from real medical reports.
- Gives a clear risk level: Low, Medium, or High.
- Explains the main reasons behind the result for that particular patient.

## 2. Approach

- Focuses on working correctly and being easy to understand, rather than being built like a large software product.
- The patient enters values from their medical reports into a simple form.
- The model predicts the risk level and shows the reasons behind the result.
- The system does not give medical advice.
- The system does not read images or scanned documents.
- The system does not act as a replacement for a doctor — it only shows the risk level and explains the prediction.

## 3. How We Are Going to Build It

- The dataset is cleaned and prepared, and the important health values are identified.
- Two models are trained on this data: a simple Logistic Regression model and a Random Forest model, and their results are compared.
- The Random Forest model is used as the main model, since it generally performs better on this type of data.
- The trained model is tested using cross-validation to make sure the result is reliable and not based on chance.
- An explanation step is added, so that for every prediction, the system also shows which health values caused that result.
- The risk levels of Low, Medium, and High are set based on the model's confidence scores, not on fixed guesswork.
- The final system is shown through a simple interface, either a Jupyter Notebook or a Streamlit page, where the patient can enter values and see the result directly, without needing a separate backend server.

## 4. How We Achieve 90% Prediction Accuracy

- Reaching 90% accuracy is not automatic — it is achieved by carefully testing and tuning the model, rather than training it once and accepting the first result.
- The model is tested using 5-fold cross-validation, so the accuracy score is reliable and not based on a single lucky split of the data.
- Model settings, such as the number of trees and the depth of each tree in the Random Forest, are tuned using grid search instead of using default values.
- The data is checked for imbalance between disease and no-disease cases, and balanced if needed.
- A few meaningful combined values, such as age together with maximum heart rate, are added to help the model recognize patterns more clearly.
- The decision threshold is adjusted so that the model correctly catches at least 90% of real disease cases, since missing a real case is more serious than raising a false alarm.
- If needed, the Logistic Regression and Random Forest models are combined as a simple ensemble to improve the result further.
- Both the accuracy score and the recall score are reported together in the final result, since correctly catching real cases matters as much as the overall accuracy number.

## 5. What We Are Going to Use in This Project

- Python — the main programming language for building and training the model.
- Pandas and NumPy — for handling and cleaning the dataset.
- Scikit-learn — for building, training, and testing the machine learning models.
- SHAP — for explaining which health values influenced each individual prediction.
- Jupyter Notebook or Streamlit — a simple interface for entering values and viewing the result, without needing a separate backend server.

## 6. Algorithms, Methods, and Models

- Logistic Regression — used as the baseline model, giving a simple starting point to compare results against.
- Random Forest — an ensemble of decision trees, used as the main model since it generally performs better on this kind of health data.
- 5-fold cross-validation — used to evaluate the model fairly on different parts of the data rather than only once.
- Grid search — used to automatically find the best model settings.
- SHAP (SHapley Additive exPlanations) — used to explain the reason behind each individual prediction.
- Threshold tuning — used to decide the cutoff points between the Low, Medium, and High risk levels, based on the model's confidence scores rather than a fixed, even split.

## 7. Types of Medical Reports We Are Going to Use

The patient enters values taken from the following reports. The first two are required for the system to make a prediction. The third is optional and improves the result further when provided.

| Report Type | Required? | What It Provides |
| --- | --- | --- |
| Lipid / Blood Report | Required | Cholesterol level and fasting blood sugar level |
| ECG Report | Required | Heart rhythm result, maximum heart rate, ST depression, ST slope, and exercise-related chest pain |
| Echocardiography Report | Optional | Number of major blood vessels seen and thalassemia result. Improves prediction confidence when available |
