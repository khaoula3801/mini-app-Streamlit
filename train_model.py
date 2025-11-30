import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# Charger ton dataset
df = pd.read_csv("CHD (1).csv")

X = df.drop("chd", axis=1)
y = df["chd"]

num_cols = ["sbp", "ldl", "adiposity", "obesity", "age"]
cat_cols = ["famhist"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(), cat_cols)
])

model = Pipeline([
    ("preprocess", preprocess),
    ("clf", RandomForestClassifier())
])

model.fit(X, y)

joblib.dump(model, "Model.pkl")
print("Model.pkl exporté avec succès !")
