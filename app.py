# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 12:33:30 2022

@author: avdhut
"""

from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.externals 
import joblib

filename = 'CRM_MODEL.pkl'
clf = pickle.load(open(filename,'rb'))
cv = pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run(debug = True)
