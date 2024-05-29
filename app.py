
from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

# importing model

model1 = pickle.load(open('class3.pkl','rb'))
model2 = pickle.load(open('reg3.pkl','rb'))
preprocessor = pickle.load(open('preprocessor3.pkl','rb'))
ms = pickle.load(open('minmaxscaler3.pkl','rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']
    soil=request.form['Soil']

    feature_list = np.array([[N, P, K, temp, humidity, ph, rainfall,soil]],dtype="object")
    transformed_features = preprocessor.transform(feature_list)
    transformed_features = ms.transform(transformed_features)
    result_crop = model1.predict(transformed_features).reshape(-1, 1)
    result_yield = model2.predict(transformed_features).reshape(-1, 1)

    crop_dict = {1: "Rice", 2: "Maize", 3: "watermelon", 4: "groundnut", 5: "cotton", 6: "banana"}

    if isinstance(result_crop, np.ndarray):
        predict_crop = result_crop[0][0]
        predict_yield = result_yield[0][0]

    if predict_crop in crop_dict:
        crop = crop_dict[predict_crop]
        result = "{} is the best crop to be cultivated with an expected yield of {} kg/ha.".format(crop, predict_yield)
    else:
        result = "Sorry, we are not able to recommend a proper crop for this environment."

    return render_template('index.html',result= result)

# python main
if __name__ == "__main__":
    app.run(debug=True)

