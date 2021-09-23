from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    AGE = request.form['age']
    JOBTYPE = request.form['JobType']
    EDUCATIONTYPE = request.form['EdType']
    MARITALSTATUS = request.form['maritalstatus']
    OCCUPATION = request.form['occupation']
    RELATIONSHIP = request.form['relationship']
    GENDER = request.form['gender']
    CAPITALGAIN = request.form['capitalgain']
    CAPITALLOSS = request.form['capitalloss']
    HOURSPERWEEK = request.form['hoursperweek']
    
    arr = np.array([[ AGE, JOBTYPE, EDUCATIONTYPE, MARITALSTATUS, 
                     OCCUPATION, RELATIONSHIP, GENDER, CAPITALGAIN, 
                     CAPITALLOSS, HOURSPERWEEK ]])
    
    arr1 = pd.DataFrame(arr, columns = ['age', 'JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'gender', 'capitalgain', 
                                          'capitalloss', 'hoursperweek'])
    pred = model.predict(arr1)
    
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)
