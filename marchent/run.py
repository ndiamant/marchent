import os
import argparse
import numpy as np
from typing import Callable, List

from marchent.marcher import (
    Marcher,
    # color rules
    rand_color,
    next_hsv,
    # move rules
    ortho_split,
)
from marchent.board import (
    draw_board,
    Board,
)


# Examples
def ortho_color() -> Board:
    h = 500
    w = 500
    marchers = [Marcher(
        w // 2, h // 2,
        color=rand_color(),
        state_transitions=np.array([
            [.9, .1, 0],  # run, split, stop
        ]),
        move_transitions=np.array([
            [1, 0, 0, 0],  # right, left, down, up
        ]),
        split_color=rand_color,
        split_move_transitions=ortho_split,
    )
    ]
    states = [0] * 100000
    return Board(
        width=w,
        height=h,
        states=states,
        marchers=marchers,
    )


def ortho_rainbow() -> Board:
    h = 500
    w = 500
    marchers = [Marcher(
        w // 2, h // 2,
        color=np.array([255, 0, 0]).astype(np.uint8),
        state_transitions=np.array([
            [.9, .1, 0],  # run, split, stop
        ]),
        move_transitions=np.array([
            [1, 0, 0, 0],  # right, left, down, up
        ]),
        split_color=next_hsv,
        split_move_transitions=ortho_split,
        )
    ]
    states = [0] * 100000
    return Board(
        width=w,
        height=h,
        states=states,
        marchers=marchers,
    )


def bacteria_rainbow() -> Board:
    h = 1000
    w = 1000
    marchers = [Marcher(
        w // 2, h // 2,
        color=np.array([0, 128, 0]).astype(np.uint8),
        state_transitions=np.array([
            [.9, .1, 0],  # run, split, stop
        ]),
        move_transitions=np.array([
            [.5, 0, .5, 0],  # right, left, down, up
        ]),
        split_color=next_hsv,
        split_move_transitions=ortho_split,
        )
    ]
    states = [0] * 100000
    return Board(
        width=w,
        height=h,
        states=states,
        marchers=marchers,
    )


EXAMPLES: List[Callable[[], Board]] = [
    ortho_color,
    ortho_rainbow,
    bacteria_rainbow,
]


if __name__ == '__main__':
    os.makedirs("outputs", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=[f.__name__ for f in EXAMPLES])
    f_name = parser.parse_args().mode
    board = next(f for f in EXAMPLES if f.__name__ == f_name)()
    draw_board(board, os.path.join("outputs", f"{f_name}.gif"), 1)
