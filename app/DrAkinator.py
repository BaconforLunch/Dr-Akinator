from model import Model

def app():
  model = Model()
  while not model.hasDeduced:
    print("Most probable diseases:")
    for dis, prob in model.diseaseProbabilities[:5]: 
      print(dis, prob)
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

if __name__ == "__main__":
  app()
  pass
