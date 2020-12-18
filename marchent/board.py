from typing import List, Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image

from marchent.marcher import (
    Marcher,
    MarcherState,
)


@dataclass
class Board:
    width: int
    height: int
    marchers: List[Marcher]
    states: List[int]

    def __post_init__(self):
        self.board = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def step(self, state: int):
        new_marchers = []
        for marcher in self.marchers:
            marcher_state = marcher.step(state)
            if marcher_state == MarcherState.STOPPING or (
                    marcher.x < 0
                    or marcher.x >= self.board.shape[1]
                    or marcher.y < 0
                    or marcher.y >= self.board.shape[0]
                    or (self.board[marcher.y, marcher.x] != 0).any()
            ):
                del marcher
                continue
            elif marcher_state == MarcherState.RUNNING:
                new_marchers.append(marcher)
            elif marcher_state == MarcherState.SPLITTING:
                new_marchers.append(marcher)
                new_marcher = marcher.split()
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
