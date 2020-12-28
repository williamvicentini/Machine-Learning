from flask import Flask,abort,request
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import json
import numpy as np
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

@app.route("/v1/previsao", methods=["POST"])
def classificar(): 
    modelo = tf.keras.models.load_model('flask_server/modelosalvo.h5')
  
    if not request.json:
       abort(400)
       
    age = request.json.get('age')
    workclass = request.json.get('workclass')
    finalweight = request.json.get('finalweight')
    education = request.json.get('education')
    educationnum = request.json.get('educationnum')
    maritalstatus = request.json.get('maritalstatus')
    occupation = request.json.get('occupation')
    relationship = request.json.get('relationship')
    race = request.json.get('race')
    sex = request.json.get('sex')
    capitalgain = request.json.get('capitalgain')
    capitalloos = request.json.get('capitalloos')
    hourperweek = request.json.get('hourperweek')
    nativecountry = request.json.get('nativecountry')

    entrada = [[age,workclass,finalweight,education,educationnum,maritalstatus,occupation,relationship,race,sex,capitalgain,capitalloos,hourperweek,nativecountry]]

    onehotencoder = load('flask_server/dicionario.joblib')
    labelencorder_saida = load('flask_server/saidadicionario.joblib')

    entradaencoded = onehotencoder.transform(entrada)

    scaler = StandardScaler(with_mean=False)

    entradaescalonada = scaler.fit_transform(entradaencoded)

    previsao = modelo.predict(entradaescalonada)
       
    return json.dumps({"valor_previsao": labelencorder_saida.inverse_transform(previsao[0].astype(int))[0]})

app.run(port = 5000, debug= False)
