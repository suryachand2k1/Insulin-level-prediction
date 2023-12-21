# Machine Learning Models for Diabetic Patient Diagnosis and Insulin Dosage Prediction

## Project Overview
In this project I am using Gradient Boosting Classifier to predict diabetes and then using Logistic Regression algorithm to predict insulin dosage in diabetic detected patients. To implement this project I am using PIMA diabetes dataset and UCI insulin dosage dataset. I am training both algorithms with above mention dataset and once after training I will upload test dataset with no class label and then Gradient Boosting will predict presence of diabetes and Logistic Regression will predict insulin dosage if diabetes detected by Gradient Boosting.

### Datasets
- **PIMA Diabetes Dataset**
- **UCI Insulin Dosage Dataset**

Datasets are available in the `Dataset` folder.

## Running the Project
1. Execute `run.bat` to start the application.
2. Upload the Diabetic Dataset using the 'Upload Diabetic Dataset' button.
3. Preprocess the dataset by removing missing values and splitting it into training and testing sets.
4. Train the Gradient Boosting algorithm and evaluate its accuracy.
5. Train the Logistic Regression algorithm and evaluate its accuracy.
6. Use the 'Predict Diabetes & Insulin Dosage' button to upload test values and receive predictions.

### Outputs
- The application displays the accuracy of both algorithms.
- It predicts the presence of diabetes and, if detected, estimates the insulin dosage.

## Requirements
- Python
- Flask (for deployment)
- Libraries: Details listed in `requirements.txt`
