import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/raw/dataset.csv")
data = data.fillna("") # remove NaNs
severity = pd.read_csv("data/raw/symptom-severity.csv")

diseases = data.Disease
symptoms = severity.Symptom

# maps string to int
diseaseEncoder = LabelEncoder()
symptomEncoder = LabelEncoder()
diseaseEncoder.fit(diseases)
symptomEncoder.fit(symptoms)  

print(data.iloc[0][data.columns[1:]])

# data = pd.read_csv("data/raw/symptom_Description.csv")
# data = pd.read_csv("data/raw/symptom_precaution.csv")
# data = pd.read_csv("data/raw/symptom-severity.csv")
# print(data.Disease)

# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=67)
