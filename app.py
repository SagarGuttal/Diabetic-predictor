from flask import Flask, render_template, request
import jsonify
import requests
import pickle
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__,template_folder="templates")
model = pickle.load(open('diabetic_classifier.pkl','rb'))

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

scaler = MinMaxScaler()

@app.route("/predict", methods=['POST'])
def predict():
    Gender = int(request.form["gender"])
    Age = int(request.form["age"])
    medical_speciality =int(request.form["medical specialty"])
    Hospital_time = int(request.form["Time in hospital"])
    dia_2 = int(request.form["Dia 2"])
    dia_3 = int(request.form["Dia 3"])
    insulin = int(request.form["Insulin"])
    metaformin = int(request.form["metformin"])
    glipizide = int(request.form["glipizide"])
    glyburide = int(request.form["glyburide"])
    pioglitazone = int(request.form["pioglitazone"])
    rosiglitazone = int(request.form["rosiglitazone"])
    glimepiride = int(request.form["glimepiride"])
    repaglinide = int(request.form["repaglinide"])
    nateglinide = int(request.form["nateglinide"])
    acarbose = int(request.form["acarbose"])
    tolazamide = int(request.form["tolazamide"])
    chlorpropamide = int(request.form["chlorpropamide"])

    final_array = [[Gender, Age, medical_speciality, Hospital_time,
                   dia_2, dia_3,insulin,metaformin,glipizide,glyburide,pioglitazone,
                   rosiglitazone,glimepiride,repaglinide,nateglinide,
                   acarbose,tolazamide,chlorpropamide]]
    scaled_array =scaler.fit_transform(final_array)
    passing_array = scaled_array.reshape(1,-1)
    final_model = model.predict(passing_array)
    if final_model ==1:
        return render_template('index.html', prediction_text = "You are diabetic")
    else:
        return render_template('index.html', prediction_text= "You are not diabetic")

if __name__ == "__main__":
    app.run(debug=True)
