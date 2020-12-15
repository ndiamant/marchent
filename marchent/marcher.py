from dataclasses import dataclass
from enum import Enum

import numpy as np


class MarcherState(Enum):
    RUNNING = 0
    SPLITTING = 1
    STOPPING = 2


@dataclass
class Marcher:
    x: int
    y: int
    move_transitions: np.ndarray  # (num global states) x 4
    state_transitions: np.ndarray  # (num global states) x (num marcher states)
    color: int  # 0-255

    def __post_init__(self):
        assert self.move_transitions.shape[1] == 4  # 4 direction choices
        assert self.state_transitions.shape[1] == len(MarcherState)

        # assert (self.move_transitions.sum(axis=1) == 1.).all()  # all valid probability distributions
        # assert (self.state_transitions.sum(axis=1) == 1.).all()

    def step(self, state: int) -> MarcherState:
        move_choice = np.random.choice(4, p=self.move_transitions[state])
        if move_choice == 0:
            self.x += 1
        elif move_choice == 1:
            self.x -= 1
        elif move_choice == 2:
            self.y += 1
        elif move_choice == 3:
            self.y -= 1

        return MarcherState(
            np.random.choice(
                len(MarcherState),
                p=self.state_transitions[state],
            )
        )
