from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from preprocessor import Preprocess
import pickle
import pandas as pd

app = Flask(__name__)
api = Api(app)


class Predict(Resource):
    def post(self):
        """
        taking taking value from the json object given by user through postman
        processing the strings of the json object
        transforming the text to extract features
        loading the saved model and predicting
        returning the json object according to the value"""
        text = request.json['sentence']
        prpobj= Preprocess()
        processed_text = prpobj.clean_data(text)
        tf1_old = pickle.load(open('vector.pkl', 'rb'))
        features = tf1_old.transform(pd.Series(processed_text))
        mnb = pickle.load(open('model1.pkl', 'rb'))
        pred = mnb.predict(features)
        return jsonify({'intent': pred[0]})


api.add_resource(Predict, '/')

if __name__ == '__main__':
    app.run(debug=True)
