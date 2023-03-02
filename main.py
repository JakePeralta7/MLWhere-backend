# Imports
from flask import request
from flask import Flask
from flask_cors import CORS, cross_origin
import os
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# Our Functions
import extract_features


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def predict_pe(file_features):
    file_array = list(file_features.values())[1:69]
    file_array = np.array([float(current_value) for current_value in file_array])
    file_array = np.expand_dims(file_array, axis=0)
    stack = joblib.load('stacking_model.pkl')
    scaler = StandardScaler()
    features = scaler.fit_transform(file_array)
    probas = stack.predict_proba(features)
    print(f"Theres a {probas[1]}% chance that this file is malware")


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/file', methods=['POST'])
@cross_origin()
def scan_file():
    try:
        file = request.files["file"]
        if file:
            file_path = os.path.join(os.getcwd(), file.filename)
            file.save(file_path)
            file_features = extract_features.extract(file_path)
            os.remove(file_path)
            if file_features == 0:
                return "Not PE"
            else:
                predict_pe(file_features)
                return "Malicious"
    except KeyError:
        return "Send me files in multipart form with the key 'file'"


def main():
    app.run(host='0.0.0.0', debug=True)


if __name__ == "__main__":
    main()
