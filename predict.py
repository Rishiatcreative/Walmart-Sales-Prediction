# import joblib
# import pandas as pd

# model = joblib.load("model/best_model.pkl")

# def predict_sales(
#     store,
#     holiday,
#     temperature,
#     fuel_price,
#     cpi,
#     unemployment,
#     year,
#     month,
#     week
# ):

#     # Ensure encoder is available; try to lazily load from encoder.pkl if not defined
#     try:
#         _ = encoder  # reference to check existence
#     except NameError:
#         import os
#         try:
#             import joblib
#         except Exception:
#             raise NameError("'encoder' is not defined and joblib is not available to load encoder.pkl")
#         encoder_path = os.path.join(os.path.dirname(__file__), "model", "encoder.pkl")
#         if os.path.exists(encoder_path):
#             encoder = joblib.load(encoder_path)
#         else:
#             raise NameError("'encoder' is not defined and encoder.pkl not found in module directory")

#     store = encoder.transform([store])[0]

#     sample = pd.DataFrame({
#         "Store": [store],
#         "Holiday_Flag": [holiday],
#         "Temperature": [temperature],
#         "Fuel_Price": [fuel_price],
#         "CPI": [cpi],
#         "Unemployment": [unemployment],
#         "Year": [year],
#         "Month": [month],
#         "Week": [week]
#     })

#     prediction = model.predict(sample)

#     return prediction[0]

import joblib
import pandas as pd

model = None
encoder = None

def load_model():
    global model, encoder

    if model is None:
        model = joblib.load("model/best_model.pkl")

    if encoder is None:
        encoder = joblib.load("model/store_encoder.pkl")

def predict_sales(store, holiday, temperature, fuel_price, cpi, unemployment, year, month, week):
    load_model()

    store = encoder.transform([store])[0]

    data = pd.DataFrame({
        "Store": [store],
        "Holiday_Flag": [holiday],
        "Temperature": [temperature],
        "Fuel_Price": [fuel_price],
        "CPI": [cpi],
        "Unemployment": [unemployment],
        "Year": [year],
        "Month": [month],
        "Week": [week]
    })

    return model.predict(data)[0]