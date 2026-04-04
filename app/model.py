import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

class Model:
  def __init__(self):
    self._data = pd.read_csv("data/raw/dataset.csv")
    # severity = pd.read_csv("data/raw/symptom-severity.csv")

    # Preprocessing
    self._data.dropna(subset=["Disease"], inplace=True) # remove no labels
    self._data = self._data.fillna("") # remove NaNs
    self._data.columns = self._data.columns.str.strip()
    self._data = self._data.map(str.strip)  # strip data of leading and trailing whitespace

    self._diseases = self._data.Disease.unique()
    self._symptoms = pd.unique(np.array([x.strip() for x in self._data[self._data.columns[1:]].values.flatten() if x]))

    # maps string to number
    self._diseaseEncoder = LabelEncoder()
    self._diseaseEncoder.fit(self._diseases)
    self._symptomEncoder = LabelEncoder()
    self._symptomEncoder.fit(self._symptoms)

    symp2dis: list[dict[str,str|float]] = []
    for _, row in self._data.iterrows():
      sample: dict[str,str|float] = {"Disease": row.Disease, **{str(symp):0 for symp in self._symptoms}}
      for symp in row[self._data.columns[1:]]:
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
          if noisy[symp] == 0:
            if rand < 0.2: noisy[symp] = 0.25
            elif rand < 0.3: noisy[symp] = 0.5
        symp2dis.append(noisy)
    X = pd.DataFrame(symp2dis)

    # Select randomly for training
    x_train, x_test, y_train, y_test = train_test_split(X.drop(columns="Disease"),X.Disease,test_size=0.4,random_state=67)

    self._clf = tree.DecisionTreeClassifier(criterion="entropy")
    self._clf = self._clf.fit(x_train,y_train)

    self.restart()
  
  def restart(self):
    self._node = 0

  @property
  def diseases(self): return self._diseases
  @property
  def symptoms(self): return self._symptoms

  @property
  def hasDeduced(self) -> bool: 
    return self._clf.tree_.children_left[self._node] == -1

  @property
  def deducedDisease(self) -> str: 
    return self._diseaseEncoder.inverse_transform([np.argmax(self._clf.tree_.value[self._node])])[0]
  
  @property
  def currentSymptom(self) -> str:
    return self._symptomEncoder.inverse_transform([self._clf.tree_.feature[self._node]])[0]
  
  @property
  def diseaseProbabilities(self):
    return sorted(
      zip(self._clf.classes_, self._clf.tree_.value[self._node][0]), 
      key = lambda kp: kp[1], 
      reverse = True,
    )

  def classify(self, score: float):
    if self.hasDeduced: return
    if self._clf.tree_.threshold[self._node] < score:
      self._node = self._clf.tree_.children_right[self._node]
    else:
      self._node = self._clf.tree_.children_left[self._node]

if __name__ == "__main__":
  model = Model()
  print(model.diseases.size)
  print(model.symptoms.size)
  print(model._data)
  pass
