# IMPORTING LIBRARIES
import pickle
import numpy as np
import pandas as pd
from CustomTransformers import (
    FeatureCreator, MasonryHandler, OrdinalMapper, GroupMapper,
    SimpleCatImputer, ValueImputer, ColumnDropper
)
from flask import Flask, request, jsonify, app, render_template, url_for

app = Flask(__name__)


# LOADING THE MODELS
Xgb_model = pickle.load(open('XGB_pipeline.pkl', 'rb'))
Elasticnet_model = pickle.load(open('Elasticnet_Pipeline.pkl', 'rb'))

# Expected features
try:
    Expected_F = pickle.load(open("ExpectedFeatures", 'rb'))
    for col in ["GarageCond", "Utilities"]:
        if col in Expected_F:
            Expected_F.remove(col)
except:
    Expected_F = None


# LOADING THE TESTING SET
test = "test.csv"




# Function to convert the data we receive into a dataframe with feature names

def to_df(data_r):
    df = pd.DataFrame(data_r)
    for col in Expected_F:
        if col not in df:
            df[col] = pd.NA
    df = df[Expected_F]
    return df

def predict_avg(X):
    p1 = Xgb_model.predict(X)
    p2 = Elasticnet_model.predict(X)
    avg = (p1 + p2) / 2
    return avg


@app.route('/')
def home():
    return render_template("home.html")

#@app.route('/predict-api',methods=['POST'])
# def predict_api():
#     data = request.json["data"]
#     print(data)
#     print(np.array(list(data.values())).reshape(1,-1))
#     xgb_result = Xgb_model.predict(np.array(list(data.values())).reshape(1,-1))
#     elasticnet_result = Elasticnet_model.predict(np.array(list(data.values())).reshape(1,-1))
#     output = (xgb_result + elasticnet_result) / 2
#     print(output)
#     return jsonify(output)

@app.route('/predict',methods=['POST',"GET"])
def predict():
    if request.method == "POST":
        data = request.form.to_dict()
        print(list(data))
        print("Expected Features:")
        print(Expected_F)

        for key, value in data.items():
            try:
                if value == "":
                    data[key] = np.nan
                elif '.' in data[key]:
                    data[key] = float(value)
                else:
                    data[key] = int(value)
            except:
                pass
        df = pd.DataFrame([data])
        print(df.columns)
        prediction = predict_avg(df)
        pred_int = int(np.round(prediction[0]))
        return render_template("result.html", predicted_price=pred_int)
    return render_template("form.html")


@app.route('/random-row',methods=['GET'])
def random_row():
    df_test = pd.read_csv('test.csv')
    row = df_test.sample(1)
    print(row)
    row = predict_avg(row)
    pred_int = int(np.round(row[0]))

    return render_template("result.html", predicted_price=pred_int)




if __name__ == "__main__":
    app.run(debug=True)
