import time
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("data/raw/dataset.csv")
# severity = pd.read_csv("data/raw/symptom-severity.csv")

# Preprocessing
data.dropna(subset=["Disease"], inplace=True) # remove no labels
data = data.fillna("") # remove NaNs
data.columns = data.columns.str.strip()
data = data.map(str.strip)  # strip data of leading and trailing whitespace

diseases = data.Disease.unique()
symptoms = pd.unique(np.array([x.strip() for x in data[data.columns[1:]].values.flatten() if x]))

# maps string to number
diseaseEncoder = LabelEncoder()
diseaseEncoder.fit(diseases)
symptomEncoder = LabelEncoder()
symptomEncoder.fit(symptoms)

# Construct dataframe
symp2dis: list[dict[str,str|float]] = []
for _, row in data.iterrows():
  sample: dict[str,str|float] = {"Disease": row.Disease, **{str(symp):0 for symp in symptoms}}
  for symp in row[data.columns[1:]]:
    if not symp: continue
    sample[symp] = 1
  
  symp2dis.append(sample)
  
  # add noisy samples
  for _ in range(10):
    noisy = sample.copy()
    for symp in noisy:
      if symp == "Disease": continue
      
      rand = np.random.random()
      if noisy[symp] == 1:
        if rand < 0.2: noisy[symp] = 0.75
        elif rand < 0.3: noisy[symp] = 0.5
      elif noisy[symp] == 0:
        if rand < 0.2: noisy[symp] = 0.25
        elif rand < 0.3: noisy[symp] = 0.5
    symp2dis.append(noisy)
X = pd.DataFrame(symp2dis)

# Select randomly for training
x_train, x_test, y_train, y_test = train_test_split(X.drop(columns="Disease"),X.Disease,test_size=0.4,random_state=67)

class Model:
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
  def deducedDisease(self) -> str: 
    return self._clf.classes_[np.argmax(self._clf.tree_.value[self._node])]
  
  @property
  def currentSymptom(self) -> str:
    return x_train.columns[self._clf.tree_.feature[self._node]]

  @property
  def diseaseProbabilities(self):
    return sorted(
      zip(self._clf.classes_, self._clf.tree_.value[self._node][0]), 
      key = lambda kp: kp[1], 
      reverse = True,
    )
  
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
  model = Model()
  # print(diseases.size)
  # print(symptoms.size)
  # print(data)
  print(model.getClassRep(x_test,y_test))
  print(f"Latency: {model.getLatency(x_test)} ms")

  mat = model.getConMat(x_test,y_test)

  plt.figure(figsize=(100,100))
  sns.heatmap(
    data        = mat,
    square      = True,
    annot       = False,
    fmt         = "d",
    cbar        = True,
    xticklabels = model.labels,   
    yticklabels = model.labels,
  )
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.title('Confusion Matrix: Predicted vs True Categories')
  plt.show()
