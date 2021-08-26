from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)


@app.route('/', methods=['POST'])
def predict():
    df = pd.read_csv('absence_grade.csv')

    def clean_grades(x):
        if x >= 50:
            return 1
        else:
            return 0

    df['grade'] = df['grade'].apply(clean_grades)
    print(df)
    X = df[['absence']]
    y = df['grade']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    logReg = LogisticRegression(solver='lbfgs', max_iter=10000)
    logReg = logReg.fit(X_train, y_train)
    y_pred = logReg.predict(X_test)
    print(y_pred)

    json_ = request.json
    query = pd.DataFrame(json_)
    print(query)
    prediction = list(logReg.predict(query))
    return jsonify({'prediction': str(prediction)})


@app.route('/hello', methods=['GET'])
def sayhello():
    return 'Hello'


@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    query = pd.DataFrame(json_)
    print(query)
    df = pd.read_csv('ProphetTest_01M019.csv')
    train_time = np.array(df['ds'])
    train_data = np.array(df['y'])
    test_data = np.array(query['y'])
    test_time = np.array(query['ds'])
    data = np.concatenate((train_data, test_data))

    split_ratio = train_data.shape[0] / data.shape[0]
    window_size = 40
    batch_size = 20
    shuffle_buffer = 1000
    split_index = int(split_ratio * data.shape[0])

    def ts_data_generator(data, window_size, batch_size, shuffle_buffer):
        ts_data = tf.data.Dataset.from_tensor_slices(data)
        ts_data = ts_data.window(window_size + 1, shift=1, drop_remainder=True)
        ts_data = ts_data.flat_map(lambda window: window.batch(window_size + 1))
        ts_data = ts_data.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
        ts_data = ts_data.batch(batch_size).prefetch(1)
        return ts_data

    tensor_train_data = tf.expand_dims(train_data, axis=-1)
    tensor_test_data = tf.expand_dims(test_data, axis=-1)

    tensor_train_dataset = ts_data_generator(tensor_train_data, window_size, batch_size, shuffle_buffer)
    tensor_test_dataset = ts_data_generator(tensor_test_data, window_size, batch_size, shuffle_buffer)

    # Fusion model of 1D CNN and LSTM
    model = tf.keras.models.Sequential([tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal",
                                                               activation="relu", input_shape=[None, 1]),
                                        tf.keras.layers.LSTM(64, return_sequences=True),
                                        tf.keras.layers.LSTM(64, return_sequences=True),
                                        tf.keras.layers.Dense(30, activation="relu"),
                                        tf.keras.layers.Dense(10, activation="relu"), tf.keras.layers.Dense(1)])
    optimizer = tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
    history = model.fit(tensor_train_dataset, epochs=30, validation_data=tensor_test_dataset)

    def model_forecast(model, data, window_size):
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.window(window_size, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size))
        ds = ds.batch(20).prefetch(1)
        forecast = model.predict(ds)
        return forecast

    rnn_forecast = model_forecast(model, data[..., np.newaxis], window_size)
    rnn_forecast = list(rnn_forecast[split_index - window_size:-1, -1, 0])

    return jsonify({'prediction': str(rnn_forecast)})






if __name__ == '__main__':
    app.run()
