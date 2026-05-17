from common import *
from typing import Sequence
from sklearn.tree import DecisionTreeClassifier

class DrAkinatorModel:
  def __init__(self, *,
    classifier: DecisionTreeClassifier,
    diseases: Sequence[Disease],
    symptoms: Sequence[Symptom],
  ):
    self._clf = classifier    # Use best tuned model
    self._diseases = {d.name:d for d in diseases}
    self._symptoms = symptoms 
    self.restart()
  
  def restart(self):
    self._node = 0

  @property
  def hasDeduced(self) -> bool: 
    return self._clf.tree_.children_left[self._node] == -1  # Check if leaf node

  @property
  def deducedDisease(self) -> Disease:
    node_vals = self._clf.tree_.value[self._node][0]
    disease = self._clf.classes_[max(range(len(node_vals)), key = lambda i: node_vals[i])]
    return self._diseases[disease]
  
  @property
  def currentSymptom(self) -> Symptom:
    return self._symptoms[self._clf.tree_.feature[self._node]]

  @property
  def diseaseProbabilities(self):
    probs = zip(self._clf.classes_, self._clf.tree_.value[self._node][0])
    return sorted(probs, key=lambda x: x[1], reverse=True)
  
  def score(self, x_test, y_test): return self._clf.score(x_test, y_test)

  def classify(self, score: GameResponse):
    if self.hasDeduced: return
    if self._clf.tree_.threshold[self._node] < score.weight:
      self._node = self._clf.tree_.children_right[self._node]
    else:
      self._node = self._clf.tree_.children_left[self._node]
