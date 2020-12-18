from dataclasses import dataclass
from typing import Callable
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
    color: np.ndarray  # (3,)

    # these are used to create a new marcher on splitting
    split_color: Callable[[np.ndarray], np.ndarray] = lambda c: c.copy()
    split_move_transitions: Callable[[np.ndarray], np.ndarray] = lambda x: x.copy()
    split_state_transitions: Callable[[np.ndarray], np.ndarray] = lambda x: x.copy()

    def __post_init__(self):
        assert self.move_transitions.shape[1] == 4  # 4 direction choices
        assert self.state_transitions.shape[1] == len(MarcherState)
        assert (self.move_transitions.sum(axis=1) == 1.).all()  # all valid probability distributions
        assert (self.state_transitions.sum(axis=1) == 1.).all()
        assert self.color.shape == (3,)  # 3 channel color

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

    def split(self) -> "Marcher":
        return Marcher(
            x=self.x, y=self.y,
            move_transitions=self.split_move_transitions(self.move_transitions),
            state_transitions=self.split_state_transitions(self.state_transitions),
            color=self.split_color(self.color),
            split_move_transitions=self.split_move_transitions,
            split_state_transitions=self.split_state_transitions,
            split_color=self.split_color,
        )


# color rules
def rand_color(c: np.ndarray=None) -> np.ndarray:
    return np.random.randint(0, 256, size=3).astype(np.uint8)


# move rules
def ortho_split(x: np.ndarray) -> np.ndarray:
    z = x.copy()
    if np.random.rand() > .5:
        z[:, 0] = x[:, 2]
        z[:, 1] = x[:, 3]
        z[:, 2] = x[:, 1]
        z[:, 3] = x[:, 0]
    else:
        z[:, 0] = x[:, 3]
        z[:, 1] = x[:, 2]
        z[:, 2] = x[:, 0]
        z[:, 3] = x[:, 1]
    return z

