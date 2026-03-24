import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

data = pd.read_csv("data/raw/dataset.csv")
data = data.fillna("") # remove NaNs
severity = pd.read_csv("data/raw/symptom-severity.csv")

symptoms = pd.unique(np.array([x.strip() for x in data[data.columns[1:]].values.flatten() if x]))

# maps string to int
diseaseEncoder = LabelEncoder()
symptomEncoder = LabelEncoder()
diseaseEncoder.fit(data.Disease)
symptomEncoder.fit(symptoms)  

symp2dis = [{symp:0 for symp in symptoms} for _ in range(len(data))] 
for i,row in data.iterrows():
  for symp in row[data.columns[1:]]:
    if not symp: continue
    symp2dis[i][symp.strip()] = 1
X = pd.DataFrame(symp2dis)

# Select randomly for training
x_train, x_test, y_train, y_test = train_test_split(X,data.Disease,test_size=0.4,random_state=67)

clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(x_train,y_train)

tree.plot_tree(clf)

def prompt(root: int = 0) -> None:
  if clf.tree_.children_left[root] == -1:
    disease = diseaseEncoder.inverse_transform([np.argmax(clf.tree_.value[root])])[0]
    print(f"You have {disease}!")
    return
  
  ans = input(f"Does your disease involve {symptoms[clf.tree_.feature[root]]} (y/n)? ").strip().lower()

  if ans == "y": return prompt(clf.tree_.children_right[root])
  if ans == "n": return prompt(clf.tree_.children_left[root])

  print("Wrong answer.")
  prompt(root)

if __name__ == "__main__":
  prompt()
