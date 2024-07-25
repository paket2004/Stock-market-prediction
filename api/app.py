from flask import Flask, request, jsonify, make_response

import mlflow
import mlflow.pyfunc
import os
import pickle
import pandas as pd

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# model = mlflow.pyfunc.load_model(os.path.join(BASE_PATH, "api", "model_dir"))
# # or
# model = mlflow.pyfunc.load_model(model_uri="/model")
# or your choice

model = mlflow.pyfunc.load_model(os.path.join(BASE_PATH, "api", "model_dir"))

# model=pickle.load(open('/home/anastasia/Downloads/Telegram Desktop/model.pkl','rb'))

app = Flask(__name__)

@app.route("/", methods = ["GET"])
def home():
	msg = """
      
	This API has two main endpoints:\n
	1. /info: to get info about the deployed model.\n
	2. /predict: to send predict requests to our deployed model.\n

	"""

	response = make_response(msg, 200)
	response.content_type = "text/plain"
	return response

@app.route("/info", methods = ["GET"])
def info():

	response = make_response(str(model.metadata), 200)
	response.content_type = "text/plain"
	return response

@app.route('/ping', methods=['GET'])
def ping():
    return "pong", 200

# /predict endpoint
@app.route("/predict", methods = ["POST"])
def predict():

    data = request.json
    # raise Exception(data)
    data = data['inputs']
    input_df = pd.DataFrame([data])
    if 'Year' in input_df.columns:
        input_df['Year'] = input_df['Year'].astype('int64')

    for column in input_df.columns:
        if 'Sector' in column:
            input_df[column] = input_df[column].astype('int64')
            
    input_df['Symbol_Index'] = input_df['Symbol_Index'].astype('int64')
	
    

    predictions = model.predict(input_df)
    return jsonify(predictions.tolist())



# This will run a local server to accept requests to the API.
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5002))
    app.run(debug=True, host='0.0.0.0', port=port)
	


