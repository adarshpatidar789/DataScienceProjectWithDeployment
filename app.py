from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            person_age = float(request.form['person_age'])
            person_income =float(request.form['person_income'])
            person_home_ownership =request.form['person_home_ownership']
            person_emp_length =float(request.form['person_emp_length'])
            loan_grade = request.form['loan_grade']
            loan_amnt =float(request.form['loan_amnt'])
            loan_int_rate =float(request.form['loan_int_rate'])
            loan_percent_income =float(request.form['loan_percent_income'])
            cb_person_default_on_file = request.form['cb_person_default_on_file']
            cb_person_cred_hist_length =float(request.form['cb_person_cred_hist_length'])
            
            # prepare DataFrame (recommended for most ML pipelines)
            data = pd.DataFrame([[person_age, person_income, person_home_ownership, person_emp_length,
                                  loan_grade, loan_amnt, loan_int_rate,
                                  loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length]],
                                columns=['person_age', 'person_income','person_home_ownership','person_emp_length',
                                        'loan_grade','loan_amnt','loan_int_rate',
                                         'loan_percent_income','cb_person_default_on_file','cb_person_cred_hist_length'])
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	
	app.run(host="0.0.0.0", port = 8080)