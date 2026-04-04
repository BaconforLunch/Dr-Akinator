import scipy

symptomSeverityFilepath = 'data/raw/Symptom-severity.csv'
datasetFilepath = 'data/raw/dataset.csv'
symptomDescriptionFilepath = 'data/raw/symptom_Description.csv'
symptomPrecautionFilepath = 'data/raw/symptom_Precaution.csv'

#filler for now
def openFile(path:str) -> list[list[str]]|None:
    try: 
        file = open(path, "r")
        res :list[list[str]]= []
        
        curLine = file.readline()
        while curLine:
            res.append([s.strip() for s in curLine.strip().split(',')])
            curLine = file.readline()
        file.close()
        return res
    except:
        print("File error")

#testing
contents = openFile(symptomDescriptionFilepath)
if contents:
    for c in contents[0:2]: print(c)
 