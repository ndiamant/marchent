import os
import argparse
import numpy as np
from typing import Callable, List

from marchent.marcher import (
    Marcher,
    rand_color,
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


EXAMPLES: List[Callable[[], Board]] = [
    ortho_color,
]


if __name__ == '__main__':
    os.makedirs("outputs", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=[f.__name__ for f in EXAMPLES])
    f_name = parser.parse_args().mode
    board = next(f for f in EXAMPLES if f.__name__ == f_name)()
    draw_board(board, os.path.join("outputs", f"{f_name}.gif"), 1)
