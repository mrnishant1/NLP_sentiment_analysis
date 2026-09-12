import os

from .inference import inference,load_model_once
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://mrnishant1.github.io"}})
load_model_once()

@app.route('/', methods=['GET'])
def home():
    return jsonify({'data': 'hello world'})

@app.route('/analyse', methods=['POST'],)
def inference_from_model():
    body = request.get_json(silent=True) or {}
    if body.get('prompt') is not None:
        result = inference(body.get('prompt'))
        return jsonify({'data': result})
    return jsonify({'data':"Invalid text input"})

# @app.route('/train',methods=['POST'])
# def start_model_training():
#     body = request.get_json(silent=True) or {}
#     if body


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
    