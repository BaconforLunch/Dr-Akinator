import os
import joblib
import pandas as pd
from common import *
from model import DrAkinatorModel
from view import DrAkinatorView, ViewObserver, DrAkinatorState, sessionState

def app(model: DrAkinatorModel):
  while not model.hasDeduced:
    print("Most probable diseases:")
    for dis, prob in model.diseaseProbabilities[:5]: 
      print(dis, prob)
    match input(f"Does your disease involve {model.currentSymptom.name} (y/py/idk/pn/n)? "):
      case "y": model.classify(GameResponse.YES)
      case "py": model.classify(GameResponse.PROBABLY_YES)
      case "idk": model.classify(GameResponse.IDK)
      case "pn": model.classify(GameResponse.PROBABLY_NOT)
      case "n": model.classify(GameResponse.NOT)
      case "reset": model.restart()
      case "quit": return
      case "exit": return
      case _: 
        print("Try again.")
        continue
  
  print(f"You have {model.deducedDisease.name}!")
  return

class DrAkinatorController(ViewObserver):
  def __init__(self, model: DrAkinatorModel, view: DrAkinatorView):
    view.addObserver(self)
    self._model = model
    self._view = view
  
  def run(self):
    disProbs = pd.DataFrame(
      data = self._model.diseaseProbabilities,
      columns = ["Disease", "Probability"],
    )

    self._view.show(DrAkinatorState(
      currentSymptom        = self._model.currentSymptom.name,
      deducedDisease        = self._model.deducedDisease,
      isGameOver            = self._model.hasDeduced,
      diseaseProbabilities  = disProbs,
    ))

  def resetGame(self):
    self._model.restart()

  def handleResponse(self, resp: GameResponse):
    self._model.classify(resp)

if __name__ == "__main__":
  payload = joblib.load(os.path.join("artifact", "model.pkl"))

  model = DrAkinatorModel(
    classifier  = payload["classifier"],
    diseases    = payload["diseases"],
    symptoms    = payload["symptoms"],
  )

  # app(model)
  if "model" not in sessionState:
    sessionState.model = model 
  view = DrAkinatorView()
  controller = DrAkinatorController(sessionState.model, view)
  controller.run()
