from dataclasses import dataclass

@dataclass(frozen=True)
class Disease:
  name: str
  description: str

