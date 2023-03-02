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
    scaler = joblib.load('scaler.pkl')
    file_array = list(file_features.values())[1:69]
    file_array = np.array([float(val) for val in file_array]).reshape(1, -1)
    file_array = scaler.transform(file_array)
    stack = joblib.load('stacking_model.pkl')
    probas = stack.predict_proba(file_array)
    print(probas)
    pred = np.argmax(np.array(probas), axis=1)[0]
    if pred == 0:
        return "Benign"
    elif pred == 1:
        return "Malicious"


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
                return predict_pe(file_features)
    except KeyError:
        return "Send me files in multipart form with the key 'file'"


def main():
    app.run(host='0.0.0.0', debug=True)


if __name__ == "__main__":
    main()
