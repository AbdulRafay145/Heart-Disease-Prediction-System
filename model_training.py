import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)

n_samples = 1000

data = {
    "age": np.random.randint(29, 77, n_samples),
    "sex": np.random.randint(0, 2, n_samples),
    "cp": np.random.randint(0, 4, n_samples),
    "trestbps": np.random.randint(90, 180, n_samples),
    "chol": np.random.randint(150, 350, n_samples),
    "fbs": np.random.randint(0, 2, n_samples),
    "thalach": np.random.randint(70, 200, n_samples),
    "exang": np.random.randint(0, 2, n_samples),
}

df = pd.DataFrame(data)

df["target"] = (
    (df["age"] > 50) &
    (df["chol"] > 240) |
    (df["trestbps"] > 150)
).astype(int)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier())
])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20]
}

grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_


y_pred = best_model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))


with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Model Saved Successfully!")