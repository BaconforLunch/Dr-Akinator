from typing import Protocol
from dataclasses import dataclass
from enum import StrEnum
from common import *
import streamlit as st
import pandas as pd

class ResponseType(StrEnum):
  YES = "Yes"
  PROBABLY_YES = "Probably"
  IDK = "Don't Know"
  PROBABLY_NOT = "Probably Not"
  NOT = "No"

sessionState = st.session_state

@dataclass(frozen=True)
class DrAkinatorState:
  currentSymptom: str
  deducedDisease: Disease
  isGameOver: bool
  diseaseProbabilities: pd.DataFrame

class ViewObserver(Protocol):
  def resetGame(self): ...
  def handleResponse(self, resp: ResponseType): ...

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
  
  def showGameOver(self, deducedDisease: Disease):
    st.success(f"You have {deducedDisease.name}!")
    st.write(deducedDisease.description)
    if st.button("Start Over"):
      for obs in self._observers: obs.resetGame()
      st.rerun()
  
  def showInquiry(self, currentSymptom: str):
    st.subheader(f"Does your disease involve {currentSymptom.replace('_', ' ')}?")
    for label in ResponseType:
      if st.button(label, use_container_width=True):
        for obs in self._observers: obs.handleResponse(label)
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
