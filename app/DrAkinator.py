from model import DrAkinatorModel, YES, PROBABLY_YES, IDK, PROBABLY_NOT, NOT
from view import DrAkinatorView, ViewObserver, DrAkinatorState, ResponseType, sessionState

def app():
  model = DrAkinatorModel()
  while not model.hasDeduced:
    print("Most probable diseases:")
    # for dis, prob in model.diseaseProbabilities[:5]: 
    #   print(dis, prob)
    match input(f"Does your disease involve {model.currentSymptom} (y/py/idk/pn/n)? "):
      case "y": model.classify(1.0)
      case "py": model.classify(0.75)
      case "idk": model.classify(0.5)
      case "pn": model.classify(0.25)
      case "n": model.classify(0)
      case "reset": model.restart()
      case "quit": return
      case "exit": return
      case _: 
        print("Try again.")
        continue
  
  print(f"You have {model.deducedDisease}!")
  return

class DrAkinatorController(ViewObserver):
  def __init__(self, model: DrAkinatorModel, view: DrAkinatorView):
    view.addObserver(self)
    self._model = model
    self._view = view
  
  def run(self):
    self._view.show(DrAkinatorState(
      currentSymptom        = self._model.currentSymptom,
      deducedDisease        = self._model.deducedDisease,
      isGameOver            = self._model.hasDeduced,
      diseaseProbabilities  = self._model.diseaseProbabilities,
    ))

  def resetGame(self):
    self._model.restart()

  def handleResponse(self, resp: ResponseType):
    match resp:
      case "Yes":
        self._model.classify(YES)
      case "Probably":
        self._model.classify(PROBABLY_YES)
      case "Don't Know":
        self._model.classify(IDK)
      case "Probably Not":
        self._model.classify(PROBABLY_NOT)
      case "No":
        self._model.classify(NOT)
    

if __name__ == "__main__":
  # app()

  if "model" not in sessionState:
    sessionState.model = DrAkinatorModel()
  view = DrAkinatorView()
  controller = DrAkinatorController(sessionState.model, view)
  controller.run()
