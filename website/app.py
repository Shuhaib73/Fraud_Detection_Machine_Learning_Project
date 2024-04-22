
from flask import Flask, render_template, request, flash, redirect, url_for, get_flashed_messages

from model import *
from model import PipelineTester
import pandas as pd
import numpy as np
import pickle 
import joblib 


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
            model_pipe = PipelineTester('binary_pipeline2.joblib', input_data)

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
    app.run()
