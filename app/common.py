from enum import Enum
from dataclasses import dataclass

class GameResponse(Enum):
  YES           = ("Yes", 1.0)
  PROBABLY_YES  = ("Probably", 0.75)
  IDK           = ("Don't Know", 0.5)
  PROBABLY_NOT  = ("Probably Not", 0.25)
  NOT           = ("No", 0.0)

  text: str
  weight: float

  def __init__(self, text: str, weight: float) -> None:
    self.text = text
    self.weight = weight

@dataclass(frozen=True)
class Disease:
  name: str
  description: str

@dataclass(frozen=True)
class Symptom:
  name: str


