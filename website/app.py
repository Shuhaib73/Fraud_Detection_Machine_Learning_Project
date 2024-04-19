
from flask import Flask, render_template, request, flash, redirect, url_for, get_flashed_messages

import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np


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
            # Load the trained RandomForest model
            with open('Model1.pkl', 'rb') as f:
                rf_model = pickle.load(f)

            # Custom transformer class for integrating SMOTE with scikit-learn pipelines.
            class SMOTETransformer:
                def __init__(self, random_state=None):
                    self.random_state = random_state
                    self.smote = SMOTE(random_state=random_state)
            
                def fit_resample(self, X, y):
                    return self.smote.fit_resample(X, y)

            # Define numerical and categorical features
            numerical_features = ['amt', 'lat', 'long', 'city_pop', 'Year', 'Hour', 'Day', 'Month', 'Age']
            
            categorical_features = ['category', 'gender', 'job']

            # Define preprocessing transformers
            numerical_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder()
            
            # Combine preprocessing steps manually
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            
            # Initialize SMOTETransformer
            smote_transformer = SMOTETransformer()

            # Preprocess the user input data
            user_preprocessed = preprocessor.transform(input_data)
            
            # Make prediction
            prediction_prob = rf_model.predict_proba(user_preprocessed)

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
    app.run()
