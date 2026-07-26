# %%
import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv("data/walmart_sales.csv")

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# %%
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

df.drop("Date", axis=1, inplace=True)

# %%
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df["Store"] = encoder.fit_transform(df["Store"])

joblib.dump(encoder, "model/store_encoder.pkl")

# %%
X = df.drop("Weekly_Sales", axis=1)

y = df["Weekly_Sales"]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# %%
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

# %%
results = {}

best_model = None
best_r2 = -100

# %%
import numpy as np

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(
        y_test,
        predictions,
        
    )
    rmse = np.sqrt(mse)
    r2 = r2_score(
        y_test,
        predictions
    )

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    print(f"\n{name}")
    print("-"*30)
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2  : {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model

# %%
joblib.dump(
    best_model,
    "model/sales_prediction_model.pkl"
)

# %%
print("\nBest Model Selected")

print(best_model)

# %%
if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print(importance)

# %%
results_df = pd.DataFrame(results).T

print(results_df)

results_df.to_csv(
    "model/model_results.csv"
)

# %%
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# %%
import numpy as np
pred = rf.predict(X_test)

print("MAE :", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R2 :", r2_score(y_test, pred))

# %%
import joblib

joblib.dump(rf, "model/best_model.pkl")

# %%



