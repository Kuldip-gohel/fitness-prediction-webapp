import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Load dataset
fitness = pd.read_csv("fitness_dataset.csv")

fitness["age_cat"] = pd.cut(
    fitness["age"], bins=[0,20,30,40,50,60,70,100],
    labels=[1,2,3,4,5,6,7]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_data, test_data in split.split(fitness, fitness["age_cat"]):
    train_set = fitness.loc[train_data].drop("age_cat", axis=1)
    test_set = fitness.loc[test_data].drop("age_cat", axis=1)

fitness = train_set.copy()

# Clean data
mean_value = fitness["weight_kg"].mean()
fitness["weight_kg"] = np.where(fitness["weight_kg"] > 160, mean_value, fitness["weight_kg"])
fitness["sleep_hours"] = fitness["sleep_hours"].fillna(fitness["sleep_hours"].median())
fitness["smokes"] = fitness["smokes"].replace({"yes":1,"Yes":1,"YES":1,"1":1,"no":0,"No":0,"NO":0,"0":0})
fitness["gender"] = fitness["gender"].map({"F":0,"M":1})

labels = fitness["is_fit"].copy()
fitness = fitness.drop("is_fit", axis=1)

num_attribs = fitness.drop("gender", axis=1).columns.tolist()
cat_attribs = ["gender"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

fitness_prepared = full_pipeline.fit_transform(fitness)

# Train model
model = LogisticRegression(random_state=42)
model.fit(fitness_prepared, labels)

# Save model + pipeline
pickle.dump(model, open("fitness_model.pkl", "wb"))
pickle.dump(full_pipeline, open("pipeline.pkl", "wb"))
