import pickle
# import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import json


app = Flask(__name__)
model = load_model('ann_model.h5')
scaler = pickle.load(open('scaler.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    newdata = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    prediction = model.predict(newdata)
    output = prediction[0].tolist()
    # print(output)
    return jsonify(output)
    # return jsonify({'prediction': output})

@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(val) for val in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    prediction = model.predict(final_input)
    output = prediction[0].tolist()
    return render_template("home.html",prediction_text = "The predicted Netflix stock value is {}".format(output[0]))

if __name__=="__main__":
    # app.run(debug = True)
    app.run(host="0.0.0.0", port=8000, debug = True)

