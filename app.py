# IMPORTING LIBRARIES
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, app, render_template, url_for

app = Flask(__name__)

# LOADING THE MODELS
Xgb_model = pickle.load(open('XGB_pipeline.pkl', 'rb'))
Elasticnet_model = pickle.load(open('Elasticnet_Pipeline.pkl', 'rb'))

# LOADING THE TESTING SET
test = "test.csv"
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict-api',methods=['POST'])
def predict_api():
    data = request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    xgb_result = Xgb_model.predict(np.array(list(data.values())).reshape(1,-1))
    elasticnet_result = Elasticnet_model.predict(np.array(list(data.values())).reshape(1,-1))
    output = (xgb_result + elasticnet_result) / 2
    print(output)
    return jsonify(output)

@app.route('/predict',methods=['POST',"GET"])
def predict():
    if request.method == "POST":
        data = request.form.to_dict()

        for key, value in data.items():
            try:
                if data[key] == "":
                    data[key] = np.nan
                elif '.' in data[key]:
                    data[key] = float(value)
                else:
                    data[key] = int(value)
            except:
                pass
        df = pd.DataFrame([data])
        prediction = int(round((Xgb_model.predict(df)+ Elasticnet_model.predict(df)) / 2))

        return render_template("result.html", predicted_price=prediction)
    return render_template("form.html")


@app.route('/random-row',methods=['GET'])
def random_row():
    df_test = pd.read_csv('test.csv')
    df_test.drop(columns=['Id'], inplace=True)
    row = df_test.sample(1)
    true_val = row["SalePrice"].values[0] if "SalePrice" in row.columns else None
    input = row.drop(columns=["SalePrice"],axis=1, errors="ignore")
    pred = int(round((Xgb_model.predict(input)+ Elasticnet_model.predict(input)) / 2))

    return render_template("result.html", predicted_price=pred, actual_price=true_val)




if __name__ == "__main__":
    app.run(debug=True)
