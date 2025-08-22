from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Lazy load models (only once at startup)
def load_resources():
    global rf, ohe, le, df, cluster_map, df_airports, feature_cols, model_code
    import flight_alternate_dates_routes_model as model_code

    rf = joblib.load("flight_price_model.pkl")
    ohe = joblib.load("onehot_encoder.pkl")
    le = joblib.load("season_labelencoder.pkl")

    df = model_code.df
    cluster_map = model_code.cluster_map
    df_airports = model_code.df_airports
    feature_cols = model_code.X_train.columns

load_resources()

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

        price = model_code.predict_price(user_input, rf, feature_cols, ohe, df)

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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



