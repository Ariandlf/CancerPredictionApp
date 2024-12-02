# Cancer Predictor App 🎗️
Welcome to the Cancer Predictor App – a powerful tool designed to predict whether a tumor is benign or malignant using machine learning. This interactive app, built with Streamlit and scikit-learn, provides real-time predictions and visualizations to help medical professionals in diagnosing breast cancer based on cytology data.

The app uses Logistic Regression to classify tumor masses from a dataset of breast cancer measurements, displaying the results in an easy-to-understand, interactive interface.

# 📌 Features
. Interactive Sidebar: Manually adjust cell measurement values using sliders to explore different predictions.

. Radar Chart Visualization: Dynamic radar chart that displays the selected input values, including mean, standard error, and worst values for various tumor characteristics.

. Real-Time Predictions: Provides instant feedback on whether the tumor is benign or malignant, along with probability scores.
Data Scaling: The model uses scaled input data to ensure predictions are more accurate.

. Medical Assistance: This app serves as a helpful tool for medical professionals, although it is not intended to replace professional judgment.

. Gradient Background Animation: Engaging and dynamic background animation for a modern user interface.

# 📁 Project Structure
This project includes the following files and directories:

Cancer-Predictor-App/
│
├── data/                # Folder containing the dataset for training
├── model/               # Folder storing trained model and scaler (.pkl files)
├── assets/              # Custom CSS styles and images for the app
│   └── style.css        # Styling for the app
├── app.py               # Streamlit app script
├── train_model.py       # Python script for training the model
├── requirements.txt     # List of dependencies for the project
└── README.md            # Project documentation


Directories and Files Explained:
. data/: Contains the CSV dataset used to train the model.
. model/: Stores the trained Logistic Regression model and scaler.
. assets/: Includes static files such as the custom CSS for styling and any image resources like screenshots.
. train_model.py: A script used to train the logistic regression model and save the trained model.
. app.py: The main file that runs the Streamlit app.


# 📊 Dataset
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Dataset, available from the UCI Machine Learning Repository.

Dataset Features:
 . The dataset contains 569 instances with 30 features for each instance.
 . Each instance represents a cell sample from a breast cancer tumor, with features like mean, standard error, and worst values for radius, texture, perimeter,      .  area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.
  
# The target variable is the diagnosis:
M = Malignant

B = Benign

Preprocessing Steps:

. Drop Unnecessary Columns: The dataset has columns like id and Unnamed: 32, which were removed.
. Target Variable Encoding: The diagnosis column is encoded as 1 for Malignant (M) and 0 for Benign (B).
. Feature Scaling: All numeric features are scaled using the StandardScaler to improve model performance.



# 🚀 Usage
1. Input Data Using Sliders:
Once the app is running, the sidebar will present multiple sliders to adjust the measurements of cell nuclei. You can input values for various tumor characteristics, such as Radius, Texture, Perimeter, Area, and more. These values will directly affect the radar chart and the prediction outcome.

2. View Radar Chart:
The radar chart will update based on the selected values, comparing the mean, standard error, and worst measurements of each characteristic for the tumor.

3. Get Predictions:
After adjusting the values, the app will predict the likelihood of the tumor being benign or malignant. The result is displayed along with probabilities.



# 📈 Model Overview
The model is a Logistic Regression classifier that predicts whether a tumor is benign or malignant based on the input features. Here's an overview of the steps involved in training the model:

Training Process:
Data Preprocessing: We clean the data by removing irrelevant columns and encoding the target variable.
Scaling: Features are scaled using StandardScaler to ensure each feature contributes equally to the model.
Model Training: We train the model using Logistic Regression, a binary classification algorithm.
Model Evaluation: The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.
The trained model and scaler are saved as .pkl files and loaded for real-time predictions in the app.

Model Evaluation Metrics:
Accuracy: Percentage of correctly classified instances.
Precision: Proportion of true positives among predicted positives.
Recall: Proportion of true positives among actual positives.
F1-Score: Harmonic mean of precision and recall.


# 🎨 User Interface
The app uses Streamlit to build an interactive user interface. It includes:

A dynamic background animation created with CSS for a modern and immersive experience.

Clean, intuitive design with Poppins font for clarity and readability.

Engaging elements such as the radar chart and real-time predictions to enhance user interaction.


# 📜 Acknowledgements
UCI Machine Learning Repository for providing the Breast Cancer Wisconsin Dataset.

Streamlit for creating an easy-to-use framework for building interactive apps.

scikit-learn for providing the Logistic Regression algorithm used for training the model.


