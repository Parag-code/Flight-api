from flask import Flask, request, jsonify
import pandas as pd
import joblib
import flight_alternate_dates_routes_model as model_code

# Load pre-trained models
rf = joblib.load("flight_price_model.pkl")
ohe = joblib.load("onehot_encoder.pkl")
le = joblib.load("season_labelencoder.pkl")

df = model_code.df
cluster_map = model_code.cluster_map
df_airports = model_code.df_airports
feature_cols = model_code.X_train.columns

app = Flask(__name__)

@app.route("/")
def home():
    return {"status": "ok"}

@app.route("/full_search", methods=["POST"])
def full_search_api():
    try:
        data = request.get_json(force=True)
        user_input = {
            "Dep_Code": data.get("Dep_Code"),
            "Arr_Code": data.get("Arr_Code"),
            "Dep_Date": data.get("Dep_Date"),
            "airline": data.get("airline", None)
        }

        # Prediction
        price = model_code.predict_price(user_input, rf, feature_cols, ohe, df)

        # Suggestions
        df_results = model_code.suggest_alternatives(
            user_input, df, rf, feature_cols, ohe, cluster_map, df_airports,
            top_k=int(request.args.get("top_k", 5)), return_df=True
        )

        return jsonify({
            "predicted_price": price,
            "suggestions": df_results.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
