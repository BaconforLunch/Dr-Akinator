import time
import pandas as pd
import numpy as np
from common import *
from enum import Enum
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

class ResponseWeight(Enum):
  YES = 1.0
  PROBABLY_YES = 0.75
  IDK = 0.5
  PROBABLY_NOT = 0.25
  NOT = 0.0

disease2SympData = pd.read_csv("data/raw/dataset.csv")
diseaseDescData = pd.read_csv("data/raw/disease-description.csv")
# severity = pd.read_csv("data/raw/symptom-severity.csv")

# Preprocessing
disease2SympData.dropna(subset=["Disease"], inplace=True) # remove no labels
disease2SympData = disease2SympData.fillna("") # remove NaNs
disease2SympData.columns = disease2SympData.columns.str.strip()
disease2SympData = disease2SympData.map(str.strip)  # strip data of leading and trailing whitespace

diseases = disease2SympData.Disease.unique()
symptoms = pd.unique(np.array([x.strip() for x in disease2SympData[disease2SympData.columns[1:]].values.flatten() if x]))

# All Diseases must have Description
assert(set(diseases).issubset(set(diseaseDescData.Disease)))


# Construct dataframe
symp2dis: list[dict[str,str|float]] = []
for _, row in disease2SympData.iterrows():
  # refactored to add instead of treating as strict yes-no, f1-score jumped for some reason
  # refactored again to add probably_yes, reasoning was that "yes" is too sure, so treat symptom as "probable" relation
  sample: dict[str,str|float] = {"Disease": row.Disease, **{str(symp):ResponseWeight.NOT.value for symp in symptoms}}
  highest = 0.0
  for symp in row[disease2SympData.columns[1:]]:
    if not symp: continue
    sample[symp] = ResponseWeight.YES.value

  symp2dis.append(sample)

  # add noisy samples
  for _ in range(10):
    noisy = sample.copy()
    for symp in noisy:
      if symp == "Disease": continue
      
      rand = np.random.random()
      if noisy[symp] == ResponseWeight.YES.value:
        if rand < 0.2: noisy[symp] = ResponseWeight.PROBABLY_YES.value
        elif rand < 0.3: noisy[symp] = ResponseWeight.IDK.value
      elif noisy[symp] == ResponseWeight.NOT.value:
        if rand < 0.2: noisy[symp] = ResponseWeight.PROBABLY_NOT.value
        elif rand < 0.3: noisy[symp] = ResponseWeight.IDK.value
    symp2dis.append(noisy)
  
# add outliers
for _ in range(10):
  symp2dis.append({"Disease": "Always Yes Syndrome", **{str(symp):ResponseWeight.YES.value for symp in symptoms}})
  symp2dis.append({"Disease": "PROBABLY YES Disease", **{str(symp):ResponseWeight.PROBABLY_YES.value for symp in symptoms}})
  symp2dis.append({"Disease": "Cluelessness", **{str(symp):ResponseWeight.IDK.value for symp in symptoms}})
  symp2dis.append({"Disease": "PROBABLY NO Disease", **{str(symp):ResponseWeight.PROBABLY_NOT.value for symp in symptoms}})
  symp2dis.append({"Disease": "Always No Syndrome", **{str(symp):ResponseWeight.NOT.value for symp in symptoms}})
diseaseDescData = pd.concat([diseaseDescData, pd.DataFrame([
  {"Disease": "Always Yes Syndrome", "Description": "Person who keeps on saying Yes. An optimist!"},
  {"Disease": "PROBABLY YES Disease", "Description": "Person who keeps on saying Probably Yes. Why so uncertain?"},
  {"Disease": "Cluelessness", "Description": "I don't know about this one."},
  {"Disease": "PROBABLY NO Disease", "Description": "Person who keeps on saying Probably Not. Why so unsure?"},
  {"Disease": "Always No Syndrome", "Description": "Person who keeps on saying No. A pessimist."},
])], ignore_index=True)

X = pd.DataFrame(symp2dis)

# Select randomly for training
x_train, x_test, y_train, y_test = train_test_split(X.drop(columns="Disease"),X.Disease,test_size=0.4,random_state=67)

class DrAkinatorModel:
  def __init__(self):
    self._clf = tree.DecisionTreeClassifier(
      criterion = "entropy",
      max_depth = 30,
    )
    self._clf = self._clf.fit(x_train,y_train)

    self.restart()
  
  def restart(self):
    self._node = 0

  @property
  def labels(self):
    return self._clf.classes_

  @property
  def hasDeduced(self) -> bool: 
    return self._clf.tree_.children_left[self._node] == -1

  @property
  def deducedDisease(self) -> Disease: 
    disease = self._clf.classes_[np.argmax(self._clf.tree_.value[self._node])]
    return Disease(
      name = disease,
      description = diseaseDescData[diseaseDescData["Disease"] == disease].Description.item(),
    )
  
  @property
  def currentSymptom(self) -> str:
    return x_train.columns[self._clf.tree_.feature[self._node]]

  @property
  def diseaseProbabilities(self):
    df = pd.DataFrame(
      data    = zip(self._clf.classes_, self._clf.tree_.value[self._node][0]), 
      columns = ["Disease", "Probability"],
    )
    return df.sort_values("Probability", ascending=False)
  
  def score(self, x_test, y_test): return self._clf.score(x_test, y_test)

  def getClassRep(self, x_test, y_test):
    y_pred = self._clf.predict(x_test)
    return classification_report(
      y_true = y_test,
      y_pred = y_pred,
      digits = 4,
    )
  
  def getConMat(self,x_test, y_test):
    y_pred = self._clf.predict(x_test)
    return confusion_matrix(
      y_true  = y_test,
      y_pred  = y_pred,
    )

  def getLatency(self,x_test):
    start = time.perf_counter()
    self._clf.predict(x_test)
    return (time.perf_counter() - start) * 1000 / x_test.shape[0]

  def classify(self, score: float):
    if self.hasDeduced: return
    if self._clf.tree_.threshold[self._node] < score:
      self._node = self._clf.tree_.children_right[self._node]
    else:
      self._node = self._clf.tree_.children_left[self._node]

if __name__ == "__main__":
  model = DrAkinatorModel()
  # print(diseases.size)
  # print(symptoms.size)
  # print(data)

  # print(model.getClassRep(x_test,y_test))
  # print(f"Latency: {model.getLatency(x_test)} ms")

  # plt.figure(figsize=(100,100))
  # sns.heatmap(
  #   data        = model.getConMat(x_test,y_test),
  #   square      = True,
  #   annot       = False,
  #   fmt         = "d",
  #   cbar        = True,
  #   xticklabels = model.labels,   
  #   yticklabels = model.labels,
  # )
  # plt.xlabel('Predicted Label')
  # plt.ylabel('True Label')
  # plt.title('Confusion Matrix: Predicted vs True Categories')
  # plt.show()
