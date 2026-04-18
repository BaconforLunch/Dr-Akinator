from typing import Protocol, Literal, cast
from dataclasses import dataclass
import streamlit as st
import pandas as pd

type ResponseType = Literal[
  "Yes",
  "Probably",
  "Don't Know",
  "Probably Not",
  "No",
]

sessionState = st.session_state

@dataclass(frozen=True)
class DrAkinatorState:
  currentSymptom: str
  deducedDisease: str
  isGameOver: bool
  diseaseProbabilities: pd.DataFrame

class ViewObserver(Protocol):
  def resetGame(self):
    ...

  def handleResponse(self, resp: ResponseType):
    ...

class DrAkinatorView:
  def __init__(self):
    self._observers: list[ViewObserver] = []

  def addObserver(self, obs: ViewObserver):
    self._observers.append(obs)

  def show(self, state: DrAkinatorState):
    self.showTitle()
    st.divider()

    if state.isGameOver:
      self.showGameOver(state.deducedDisease)
    else:
      self.showInquiry(state.currentSymptom)
      st.divider()
      self.showDiseaseProbabilities(state.diseaseProbabilities)

  def showTitle(self):
    st.title("Dr Akinator")
  
  def showGameOver(self, deducedDisease: str):
    st.success(f"You have {deducedDisease}!")
    if st.button("Start Over"):
      for obs in self._observers: obs.resetGame()
      st.rerun()
  
  def showInquiry(self, currentSymptom: str):
    st.subheader(f"Does your disease involve {currentSymptom}?")
    for label in [
      "Yes",
      "Probably",
      "Don't Know",
      "Probably Not",
      "No",
    ]:
      if st.button(label, use_container_width=True):
        for obs in self._observers: obs.handleResponse(cast(ResponseType, label))
        st.rerun()

    if st.button("Reset"):
      for obs in self._observers: obs.resetGame()
      st.rerun()
  
  def showDiseaseProbabilities(self, diseaseProbabilities: pd.DataFrame):
    st.subheader("Most probable diseases:")
    st.bar_chart(
      data        = diseaseProbabilities,
      x           = "Disease", 
      y           = "Probability",
      horizontal  = True,
      sort        = False,
    )
