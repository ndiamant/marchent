from dataclasses import dataclass
from typing import Callable, List
from enum import Enum
import colorsys

import numpy as np


class MarcherState(Enum):
    RUNNING = 0
    SPLITTING = 1
    STOPPING = 2


def assert_valid_distribution(x: np.ndarray):
    np.testing.assert_allclose(x.sum(axis=1), 1)


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
        assert_valid_distribution(self.move_transitions)
        assert_valid_distribution(self.state_transitions)
        assert self.color.shape == (3,)  # 3 channel color
        assert not (self.color == 0).all()  # can't be collision color

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

    @property
    def expected_angle(self) -> np.ndarray:
        """expected movement angle per global state measured in degrees"""
        return np.arctan2(
            self.move_transitions[:, 3] - self.move_transitions[:, 2],
            self.move_transitions[:, 0] - self.move_transitions[:, 1]
        ) / (2 * np.pi) * 360


# color rules
def rand_color(c: np.ndarray = None) -> np.ndarray:
    return np.random.randint(0, 256, size=3).astype(np.uint8)


def next_hsv(c: np.ndarray) -> np.ndarray:
    h, s, v = colorsys.rgb_to_hls(c[0] / 255, c[1] / 255, c[2] / 255)
    h += .05
    h %= 1
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (np.array([r, g, b]) * 255).astype(np.uint8)


def lighten(c: np.ndarray) -> np.ndarray:
    H, L, S = colorsys.rgb_to_hls(c[0] / 255, c[1] / 255, c[2] / 255)
    L += .01 * (1 - L)
    r, g, b = colorsys.hls_to_rgb(H, L, S)
    return np.clip((np.array([r, g, b]) * 255).astype(np.uint8), 1, 255)


def turn_color(c: np.ndarray, next_colors: List[np.ndarray]) -> np.ndarray:
    return next_colors[np.random.randint(len(next_colors))]


# move rules
def ortho_split(x: np.ndarray) -> np.ndarray:
    z = x.copy()
    if np.random.rand() > .5:
        # right, left, down, up
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


def angle_split(x: np.ndarray, theta: float) -> np.ndarray:
    ortho = ortho_split(x)
    return np.sin(theta)**2 * ortho + np.cos(theta)**2 * x
