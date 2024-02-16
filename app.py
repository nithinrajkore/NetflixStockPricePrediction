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
    newdata = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    prediction = model.predict(newdata)
    output = prediction[0].tolist()
    # print(output)
    return jsonify(output)
    # return jsonify({'prediction': output})

if __name__=="__main__":
    app.run(debug = True)
