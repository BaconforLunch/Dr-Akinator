import streamlit as st
from model import Model

def app():
  model = Model()
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

def stapp():
  if "model" not in st.session_state:
    st.session_state.model = Model()

  model = st.session_state.model

  st.title("Dr Akinator")

  if model.hasDeduced:
    st.success(f"You have {model.deducedDisease}!")
    if st.button("Start Over"):
      model.restart()
      st.rerun()
  else:    
    st.divider()
    st.subheader(f"Does your disease involve {model.currentSymptom}?")
    for label, val in {
      "Yes": 1.0,
      "Probably": 0.75,
      "Don't Know": 0.5,
      "Probably Not": 0.25,
      "No": 0,
    }.items():
      if st.button(label, use_container_width=True):
        model.classify(val)
        st.rerun()

    if st.button("Reset"):
      model.restart()
      st.rerun()

    st.divider()
    st.subheader("Most probable diseases:")
    st.bar_chart(
      data        = model.diseaseProbabilities, 
      x           = "Disease", 
      y           = "Probability",
      horizontal  = True,
      sort        = False,
    )

if __name__ == "__main__":
  # app()
  stapp()  
