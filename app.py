from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

openmodel = open("loan.pkl", "rb")
ml_model = joblib.load(openmodel)


@app.route('/')
def Home():
    return render_template('home.html')


@app.route('/form')
def Form():
    return render_template('form.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            Gender = float(request.form['Gender'])
            Married = float(request.form['Married'])
            Dependents = float(request.form['Dependents'])
            Education = float(request.form['Education'])
            Self_Employed = float(request.form['Self_Employed'])
            Applicant_Income = float(request.form['Applicant_Income'])
            Coapplicant_Income = float(request.form['Coapplicant_Income'])
            Loan_Amount = float(request.form['Loan_Amount'])
            Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
            Credit_History = float(request.form['Credit_History'])
            Property_Area = float(request.form['Property_Area'])

            pred_args = [Gender, Married, Dependents, Education, Self_Employed, Applicant_Income, Coapplicant_Income,
                         Loan_Amount, Loan_Amount_Term, Credit_History, Property_Area]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)

            # openmodel = open("DecisionTreeModel.pkl","rb")
            # ml_model=joblib.load(openmodel)
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction))
            if model_prediction == 1:
                result = "Welcome !!! You are Eligible"
            else:
                result = "Sorry!!! You are not Eligible For  the Loan"

        except ValueError:
            result = "Please, enter Details."
        except Exception:
            result = "Please, check the details you entered."
    return render_template('Predict.html', prediction=result)


if __name__ == "__main__":
    app.run(debug=True)





















if __name__=='__main__':
    app.run(debug = True)




