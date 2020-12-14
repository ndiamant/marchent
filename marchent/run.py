import os
import numpy as np

from marchent.marcher import (
    Marcher,
    MarcherState,
)
from marchent.board import (
    draw_board,
    Board,
)


if __name__ == '__main__':
    h = 500
    w = 500
    n = 2
    marchers = [Marcher(
        w // n * i, w // n * j,
        color=int(255 * (i * j) / n**2),
        state_transitions=np.array([
            [.4, .5, .1],  # run, split, stop
        ]),
        move_transitions=np.array([
            [.25, .25, .25, .25],  # left, right, down, up
        ]),
        )
        for i in range(1, n) for j in range(1, n)
    ]
    states = [0] * 10000
    board = Board(
        width=w,
        height=h,
        states=states,
        marchers=marchers,
    )
    os.makedirs("outputs", exist_ok=True)
    draw_board(board, os.path.join("outputs", "test.gif"), 2)
