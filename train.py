import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

from preprocess import load_data, clean_data

# Load and clean data
df = load_data("data/housing.csv")
df = clean_data(df)

# Keep only useful columns
df_model = df[['location', 'total_sqft', 'bath', 'bhk', 'price']]

# One-hot encode location
df_model = pd.get_dummies(df_model, columns=['location'], drop_first=True)

# Split features and target
X = df_model.drop('price', axis=1)
y = df_model['price']

# Save training columns for prediction
with open("columns.json", "w") as f:
    json.dump(X.columns.tolist(), f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0)
}

best_model = None
best_r2 = float("-inf")
best_name = None

print("Training completed")
print("X shape:", X.shape)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name}:")
    print("MSE:", mse)
    print("R2 Score:", r2)

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name

# Save best model
joblib.dump(best_model, "model.pkl")

print(f"\nBest model saved: {best_name}")
print(f"Best R2 Score: {best_r2}")