from flask import Flask, make_response, request, jsonify, url_for, render_template
import io
import csv
import pandas as pd
from celery import Celery
from Util.celery_def import make_celery
import random
import time



app = Flask(__name__)

from Definitions.model_definitions import models
from Definitions.connectors import connectors
from Definitions.transformers import transformers
from Definitions.estimators import estimators
from Util.get_model_to_train import get_model


@app.route('/')
def index():
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/get_models', methods=['GET'])
def get_models():
    return jsonify({"models": models})

@app.route('/get_nodes', methods=['GET'])
def get_nodes():
    nodes = {
        'connector': connectors,
        'transformer': transformers,
        'estimator': estimators
    }
    response = jsonify(nodes)
    # temporary solution for CORS
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/train_model/<string:learning_type>/<string:model_category>/<string:model>', methods=['POST'])
def train_model(learning_type, model_category, model):
    print(learning_type, ", ", model_category, ". ", model)

    print(request.json)

    X = request.json['training_data'][0]['X']
    Y = request.json['training_data'][0]['Y']

    print(X, Y)

    model = get_model(model)

    print(model)

    model.fit(X, Y)

    print(model)

    m = model.coef_[0]
    b = model.intercept_
    result = ' y = {0} * x + {1}'.format(m, b)
    print(result)

    return jsonify({'trained_model': result}), 201


@app.route('/upload_csv')
def form():
    return """
        <html>
            <body>
                <h1>Transform a file demo</h1>

                <form action="/transform" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" />
                    <input type="submit" />
                </form>
            </body>
        </html>
    """

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")

@app.route('/transform', methods=["POST"])
def transform_view():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)

    print("Panda")
    df = pd.read_csv(stream)
    print(df)

    csv_input = csv.reader(stream)

    print(csv_input)

    # for row in csv_input:
    #     print(row)

    stream.seek(0)
    result = transform(stream.read())
    response = make_response(result)
    response.headers["Content-Disposition"] = "attachment; filename=umitsresult.csv"
    return response

if __name__ == '__main__':
    app.run()
