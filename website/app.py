
from flask import Flask, render_template, request, flash, redirect, url_for, get_flashed_messages
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

# Importing modules for data preprocessing and model evaluation

import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin


# Custom transformer class for integrating SMOTE (Synthetic Minority Over-sampling Technique) with scikit-learn pipelines.

class SMOTETransformer(TransformerMixin):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.smote = SMOTE(random_state=random_state)

    def fit(self, X, y=None):
        # SMOTE should not be fit during the training phase
        return self

    def transform(self, X, y=None):
        if y is None:
            # Return input data unchanged if target labels are not provided (e.g., during transformation)
            return X
        else:
            # Apply SMOTE transformation during training phase
            X_resampled, y_resampled = self.smote.fit_resample(X, y)
            return X_resampled, y_resampled


class PipelineTester:
  def __init__(self, pipeline_path: str, test_data: pd.DataFrame):
    self.pipeline_path = pipeline_path
    self.test_data = test_data

  def predict(self):
    with open(self.pipeline_path, 'rb') as file:
        loaded_pipeline = pickle.load(file)

        # Get the probability scores for each class
        probability_scores = loaded_pipeline.predict_proba(self.test_data)

        # Accessing the probability for fraudulent transaction
        fraud_class_prob = probability_scores[:, 1]
        
        return fraud_class_prob



app = Flask(__name__)

app.config['SECRET_KEY'] = 'abcdef'

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        category = request.form['category']
        gender = request.form['gender']
        Age = int(request.form['Age'])
        Job = request.form['job']
        Amt = float(request.form['amt'])
        Year = int(request.form['Year'])
        lat = float(request.form['lat'])
        log = float(request.form['long'])
        City_Pop = float(request.form['city_pop'])
        Month = int(request.form['Month'])
        Day = int(request.form['Day'])
        Hour = int(request.form['Hour'])

        # Creating a DataFrame with the input data
        input_data = pd.DataFrame({
            'category': [category],
            'amt': [Amt],
            'gender': [gender],
            'lat': [lat],
            'long': [log],
            'city_pop': [City_Pop],
            'job': [Job],
            'Year': [Year],
            'Hour': [Hour],
            'Day': [Day],
            'Month': [Month],
            'Age': [Age]
        })
        

        try:
            # Initialize the PipelineTester with the trained pipeline and input data
            model_pipe = PipelineTester('binary_pipeline1.pkl', input_data)

            # Predict the class label for the input data
            prediction_prob = model_pipe.predict()

            # Set a threshold for classification
            threshold = 0.5

            # Apply a threshold to classify the sample
            if prediction_prob > threshold:
                msg = "This is a Fraudulent Transaction"
                flash(msg, category='error')
            else:
                msg="Legitimate Transaction"
                flash(msg, category='success')

        except Exception as e:
            flash(f"Prediction Failed: {str(e)}", category='error')

    return render_template('dashboard.html')



@app.route('/about')
def about():
    
    return render_template('about.html')

@app.route('/contact')
def contact():

    return render_template('contact.html')


if __name__ == "__main__":
    app.run(debug=True)
