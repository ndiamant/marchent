from typing import List, Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image

from marchent.marcher import (
    Marcher,
    MarcherState,
)


def copy_marcher(marcher: Marcher) -> Marcher:
    return Marcher(**marcher.__dict__)


def copy_darker(marcher: Marcher) -> Marcher:
    new_marcher = Marcher(**marcher.__dict__)
    new_marcher.color = max(int(new_marcher.color * .9), 1)
    return new_marcher


def entropy_copy(marcher: Marcher) -> Marcher:
    new_marcher = Marcher(**marcher.__dict__)
    new_marcher.color = max(new_marcher.color - np.random.choice([0, 1], p=[.5, .5]), 1)
    entropy = .0025
    new_marcher.move_transitions = (1 - entropy) * new_marcher.move_transitions + entropy / 2  # moves all towards .5
    new_marcher.move_transitions /= new_marcher.move_transitions.sum(axis=1)[:, np.newaxis]
    return new_marcher


def ortho_marcher(marcher: Marcher) -> Marcher:
    new_marcher = Marcher(**marcher.__dict__)
    x = marcher.move_transitions.copy()
    if np.random.rand() > .5:
        x[:, 0] = marcher.move_transitions[:, 2]
        x[:, 1] = marcher.move_transitions[:, 3]
        x[:, 2] = marcher.move_transitions[:, 1]
        x[:, 3] = marcher.move_transitions[:, 0]
    else:
        x[:, 0] = marcher.move_transitions[:, 3]
        x[:, 1] = marcher.move_transitions[:, 2]
        x[:, 2] = marcher.move_transitions[:, 0]
        x[:, 3] = marcher.move_transitions[:, 1]
    new_marcher.move_transitions = x
    new_marcher.color = max(int(new_marcher.color * .9), 1)
    return new_marcher


def perturbed_ortho_marcher(marcher: Marcher) -> Marcher:
    new_marcher = Marcher(**marcher.__dict__)
    x = marcher.move_transitions.copy()
    if np.random.rand() > .5:
        x[:, 0] = marcher.move_transitions[:, 2]
        x[:, 1] = marcher.move_transitions[:, 3]
        x[:, 2] = marcher.move_transitions[:, 1]
        x[:, 3] = marcher.move_transitions[:, 0]
    else:
        x[:, 0] = marcher.move_transitions[:, 3]
        x[:, 1] = marcher.move_transitions[:, 2]
        x[:, 2] = marcher.move_transitions[:, 0]
        x[:, 3] = marcher.move_transitions[:, 1]
    perturb = .2
    x += perturb / 2 - np.random.rand(*x.shape) * perturb
    x = np.clip(x, 0, 1)
    x /= x.sum(axis=1)[:, np.newaxis]
    new_marcher.move_transitions = x
    new_marcher.color = max(int(new_marcher.color * .9), 1)
    return new_marcher


@dataclass
class Board:
    width: int
    height: int
    marchers: List[Marcher]
    states: List[int]
    split_marcher: Callable[[Marcher], Marcher] = copy_marcher

    def __post_init__(self):
        self.board = np.zeros((self.height, self.width), dtype=np.uint8)

    def step(self, state: int):
        new_marchers = []
        for marcher in self.marchers:
            marcher_state = marcher.step(state)
            if marcher_state == MarcherState.STOPPING or (
                    marcher.x < 0
                    or marcher.x >= self.board.shape[1]
                    or marcher.y < 0
                    or marcher.y >= self.board.shape[0]
                    or self.board[marcher.y, marcher.x] != 0
            ):
                del marcher
                continue
            elif marcher_state == MarcherState.RUNNING:
                new_marchers.append(marcher)
            elif marcher_state == MarcherState.SPLITTING:
                new_marchers.append(marcher)
                new_marcher = self.split_marcher(marcher)
                new_marcher.step(state)
                new_marchers.append(new_marcher)
            else:
                raise NotImplementedError(f"Unhandled marcher state {marcher_state}")
            self.board[marcher.y, marcher.x] = marcher.color  # draw marcher's new position
        self.marchers = new_marchers

    def run(self) -> np.ndarray:
        for i, state in enumerate(self.states):
            n_marchers = len(self.marchers)
            if n_marchers == 0:
                break
            self.step(state)
            status = f"{i + 1} / {len(self.states)} [{n_marchers}]"
            yield self.board.copy()
            print(status, end="\r")


def draw_board(board: Board, path: str, upsample: int = 1):
    resize = (board.width * upsample, board.height * upsample)  # TODO: not sure if width or height first
    images = [
        Image.fromarray(x).resize(resize, Image.BOX)
        for x in board.run()
    ]
    with open(path, 'wb') as f:
        images[0].save(f, save_all=True, append_images=images[1:] + 50 * [images[-1]], duration=50, loop=0)
