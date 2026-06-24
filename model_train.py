import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# تقييم الموديل
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import pandas as pd

df = pd.read_csv(r'G:\c++\Colab Notebooks\cv\resume_data.csv')
df.head()
df.info(),df.describe()
df.duplicated().sum()
print((df.isna().sum() / len(df)) * 100)
df.fillna('', inplace=True)
for col in df.columns:
    df[col] = (df[col].astype(str)
                      .str.replace('[', '', regex=False)
                      .str.replace(']', '', regex=False)
                      .str.replace("'", '', regex=False))
df.replace(['nan', 'None', 'N/A', 'N/A, N/A'], 0, inplace=True)
df.head()
num_col = [col for col in df.columns if df[col].dtype != object]
cat_col = [col for col in df.columns if df[col].dtype == object]
df["text"] = (
    df["career_objective"].astype(str) + " " +
    df["skills"].astype(str) + " " +
    df["educational_institution_name"].astype(str) + " " +
    df["degree_names"].astype(str) + " " +
    df["major_field_of_studies"].astype(str) + " " +
    df["professional_company_names"].astype(str) + " " +
    df["responsibilities"].astype(str) + " " +
    df["skills_required"].astype(str) + " " +
    df["educationaL_requirements"].astype(str) + " " +
    df["﻿job_position_name"].astype(str)
)
X=df['text']
y=df['matched_score']
#تحويل النص لأرقام (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer

TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    min_df=2
)
from sklearn.model_selection import train_test_split

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    min_df=2
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

import joblib

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")