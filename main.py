from flask import Flask, render_template, request
import pandas as pd
import boto3
import pickle

from preprocessing import data_preprocessor
s3 = boto3.client('s3')
response = s3.get_object(
    Bucket = "nifo-bucket",
    Key = "Telco_model_colab.pkl"
)

application = Flask(__name__)
app = application

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        form_data = pd.DataFrame(
            {"gender":[request.form.get("gender")],
             "SeniorCitizen":[float(request.form.get("senior_citizen"))],
             "Partner":[request.form.get("Partner?")],
             "Dependents": [request.form.get("Dependents?")],
             "tenure": [float(request.form.get("Length of Customership"))],
             "PhoneService": [request.form.get("PhoneService?")],
             "MultipleLines":[request.form.get("Multiple_Lines")],
             "InternetService": [request.form.get("Internet_Service")],
             "OnlineSecurity":[request.form.get("Online_Security")],
             "OnlineBackup":[request.form.get("Online_backup")],
             "DeviceProtection":[request.form.get("Device_protection")],
             "TechSupport":[request.form.get("Tech_support")],
             "StreamingTV":[request.form.get("Streaming_TV?")],
             "StreamingMovies":[request.form.get("Streaming_Movies?")],
             "Contract":[request.form.get("Contract_Length")],
             "MonthlyCharges":[float(request.form.get("Monthly_Charges"))],
             "TotalCharges":[float(request.form.get("TotalCharges"))]


            }
        )
        prediction = model.predict(form_data)[0]
    return render_template("main.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
