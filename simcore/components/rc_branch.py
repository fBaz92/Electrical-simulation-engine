from __future__ import annotations
from dataclasses import dataclass
from .base import CompositeBranchComponent
from .resistor import Resistor
from .capacitor import Capacitor


@dataclass
class SeriesRC(CompositeBranchComponent):
    """
    Two-terminal RC network with a resistor in series with a capacitor.

    This demonstrates how to create composite components: internally the branch
    is expanded into the primitive resistor/capacitor pair, yet externally it
    behaves like any other bipole.
    """
    R: float
    C: float

    def __post_init__(self) -> None:
        super().__init__()
        self.add_internal_node("mid")
        self.add_branch("R", self.POSITIVE_NODE, "mid", Resistor(self.R))
        self.add_branch("C", "mid", self.NEGATIVE_NODE, Capacitor(self.C))
