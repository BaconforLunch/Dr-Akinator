import os
import time
import joblib
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.model_selection import train_test_split, StratifiedKFold # type: ignore
from sklearn.metrics import classification_report, confusion_matrix   # type: ignore
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import seaborn as sns
import matplotlib.pyplot as plt

from common import *

SEED = 67

DATA_DIR = "data/raw"
ARTIFACT_DIR = "artifact"

disease2SympData = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
diseaseDescData = pd.read_csv(os.path.join(DATA_DIR, "disease-description.csv"))
# severity = pd.read_csv(os.path.join(DATA_DIR, "symptom-severity.csv"))

# Preprocessing
disease2SympData.dropna(subset=["Disease"], inplace=True) # remove no labels
disease2SympData = disease2SympData.fillna("") # remove NaNs
disease2SympData.columns = disease2SympData.columns.str.strip()
disease2SympData = disease2SympData.map(str.strip)  # strip data of leading and trailing whitespace

diseases = disease2SympData.Disease.unique()
symptoms = pd.unique(np.array([x.strip() for x in disease2SympData[disease2SympData.columns[1:]].values.flatten() if x]))

# All Diseases must have Description
if not set(diseases).issubset(set(diseaseDescData.Disease)):
  raise Exception("Not all diseases were described.")

# Construct dataframe
symp2dis: list[dict[str,str|float]] = []
for _, row in disease2SympData.iterrows():
  # Encode samples hotly
  sample: dict[str,str|float] = {
    "Disease": row.Disease, 
    **{str(symp):GameResponse.NOT.weight for symp in symptoms},
  }
  for symp in row[disease2SympData.columns[1:]]:
    if not symp: continue
    sample[symp] = GameResponse.YES.weight
  symp2dis.append(sample)
df = pd.DataFrame(symp2dis)

# Aggregate rows by disease
df = df.groupby("Disease", as_index=False).max()

x_train, y_train = df.drop(columns="Disease"), df.Disease
x_test, y_test = x_train, y_train
# # Split dataset
# x_train, x_test, y_train, y_test = train_test_split(
#   df.drop(columns="Disease"), 
#   df.Disease, 
#   test_size=0.25, 
#   random_state=SEED,
#   stratify=df.Disease,
# )

# Duplicate training data
x_train = pd.concat([x_train]*5, ignore_index=True)
y_train = pd.concat([y_train]*5, ignore_index=True)

# # Augment duplicates with random noise
# noise_mask = (x_train == GameResponse.YES.weight) & (np.random.rand(*x_train.shape) < 0.15)
# x_train = pd.DataFrame(
#   data = np.where(noise_mask, GameResponse.NOT.weight, x_train),
#   columns = x_train.columns,
#   index = x_train.index,
# )

# add outliers
outlier_features: list[dict[str,str|float]] = []
outlier_labels: list[str] = []
for _ in range(5):
  outlier_features.append({str(symp): GameResponse.YES.weight for symp in symptoms})
  outlier_labels.append("Always Yes Syndrome")
  outlier_features.append({str(symp): GameResponse.PROBABLY_YES.weight for symp in symptoms})
  outlier_labels.append("Probably Yes Disease")
  outlier_features.append({str(symp): GameResponse.IDK.weight for symp in symptoms})
  outlier_labels.append("Cluelessness")
  outlier_features.append({str(symp): GameResponse.PROBABLY_NOT.weight for symp in symptoms})
  outlier_labels.append("Probably No Disease")
  outlier_features.append({str(symp): GameResponse.NOT.weight for symp in symptoms})
  outlier_labels.append("Always No Syndrome")
x_train = pd.concat([x_train, pd.DataFrame(outlier_features)], ignore_index = True)
y_train = pd.concat([y_train, pd.Series(outlier_labels)], ignore_index = True)

feature_names = list(x_train.columns)

# add outliers' description
diseaseDescData = pd.concat([diseaseDescData, pd.DataFrame([
  {"Disease": "Always Yes Syndrome", "Description": "Person who keeps on saying Yes. An optimist!"},
  {"Disease": "Probably Yes Disease", "Description": "Person who keeps on saying Probably Yes. Why so uncertain?"},
  {"Disease": "Cluelessness", "Description": "I don't know about this one."},
  {"Disease": "Probably No Disease", "Description": "Person who keeps on saying Probably Not. Why so unsure?"},
  {"Disease": "Always No Syndrome", "Description": "Person who keeps on saying No. A pessimist."},
])], ignore_index=True)

# Create pipeline
pipeline = Pipeline(steps = [
  ("clf", tree.DecisionTreeClassifier(random_state=SEED, class_weight="balanced")),
])  

# Define hyperparameters for tuning
param_grid = {
  "clf__criterion": ["gini", "entropy", "log_loss"],
  "clf__max_depth": [*range(28,35), None],
  "clf__min_samples_split": [2, 3, 4],    # Min samples per internal node
  "clf__min_samples_leaf": [1,2],   # Min samples per leaf
}

# Tune model
gscv = GridSearchCV(
  estimator   = pipeline,
  param_grid  = param_grid,
  scoring     = "f1_macro",
  cv          = StratifiedKFold(n_splits = 5, shuffle = True, random_state = SEED), # use 5 folds
  n_jobs      = -1,                                                                 # use all processors
)
gscv.fit(x_train, y_train)

best_classifier = gscv.best_estimator_.named_steps["clf"]

def save_model():
  ARTIFACT_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")
  print(f"Saving best model into {ARTIFACT_PATH}...")
  os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
  payload = {
    "classifier": best_classifier,
    "symptoms": [Symptom(name=symp) for symp in feature_names],
    "diseases": [Disease(name=row["Disease"], description=row["Description"]) for _, row in diseaseDescData.iterrows()]
  }
  
  joblib.dump(payload, ARTIFACT_PATH)
  print("🎉 Storage complete! Ready for application use.")  

if __name__ == "__main__":
  save_model()
  
  print(f"Diseases: {diseases.size} | Symptoms: {symptoms.size}")
  print(disease2SympData)

  # Test model
  start = time.perf_counter()
  y_pred = best_classifier.predict(x_test)
  latency = (time.perf_counter() - start) * 1000 / x_test.shape[0]

  test_labels = np.unique(y_test)
  print(classification_report(
    y_true = y_test,
    y_pred = y_pred,
    labels = test_labels,
    digits = 4,
  ))  # type: ignore

  print(f"Latency: {latency} ms")
  print(f"Best parameters: {gscv.best_params_}")
  print(f"Best cross-validation score: {gscv.best_score_:.3f}")

  plt.figure(figsize=(100,100))
  sns.heatmap(
    data        = confusion_matrix(
      y_true  = y_test,
      y_pred  = y_pred,
    ),
    square      = True,
    annot       = False,
    fmt         = "d",
    cbar        = True,
    xticklabels = best_classifier.classes_,   
    yticklabels = best_classifier.classes_,
  )
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.title('Confusion Matrix: Predicted vs True Categories')
  plt.show()
