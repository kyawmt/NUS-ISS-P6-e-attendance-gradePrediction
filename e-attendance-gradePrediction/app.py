
from flask import Flask, jsonify, request
import pickle as p
import pandas as pd


app = Flask(__name__)



@app.route('/predict', methods=['POST'])
def predict():
        json_ = request.json
        query = pd.DataFrame(json_)
        logReg = p.load(open('final_prediction.pickle', 'rb'))
        prediction = logReg.predict(query)
        print(prediction)
        return jsonify({'prediction': str(prediction)})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
