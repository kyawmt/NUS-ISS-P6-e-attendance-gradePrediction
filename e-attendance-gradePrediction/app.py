
from flask import Flask, jsonify, request
import pickle as p
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv('absence_grade.csv')
df.loc[df['grade']<50,'grade']=0
df.loc[df['grade']>=50,'grade']=1
X=df[['absence']]
y = df['grade']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state =42)
logReg = LogisticRegression(solver = 'lbfgs', max_iter=10000)
logReg = logReg.fit(X_train, y_train)
y_pred = logReg.predict(X_test)
print(y_pred)
print(accuracy_score(y_test, y_pred))
pickle.dump(logReg,open('final_prediction.pickle','wb'))

app = Flask(__name__)



@app.route('/predict', methods=['POST'])
def predict():
        json_ = request.json
        query = pd.DataFrame(json_)
        logReg = p.load(open('final_prediction.pickle', 'rb'))
        prediction = logReg.predict(query)
        print(prediction)
        return jsonify({'prediction': str(prediction)})
