from flask import Flask,abort,request
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import json
import numpy as np

app = Flask(__name__)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@app.route("/v1/previsao", methods=["POST"])
def classificar(): 
    modelo = tf.keras.models.load_model('flask_server/modelosalvo.h5')
  
    if not request.json:
       abort(400)
       
    income = request.json.get('income')
    age = request.json.get('age')
    loan = request.json.get('loan')
    
    array = [income,age,loan]

    scaler = StandardScaler()

    predicao = modelo.predict(scaler.fit_transform([array]))
       
    return json.dumps({"valor_previsao": predicao[0][0]}, cls=NumpyEncoder)

app.run(port = 5000, debug= False)