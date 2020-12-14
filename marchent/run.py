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
    n = 5
    marchers = [Marcher(
        w // n * i, w // n * j,
        color=int(255 * (i * j) / n**2),
        state_transitions=np.array([
            [.4, .5, .1],  # run, split, stop
            [.4, .5, .1],
            [.4, .5, .1],
            [.4, .5, .1],
        ]),
        move_transitions=np.array([
            [.0, .5, .5, .0],  # left, right, down, up
            [.5, .0, .5, .0],
            [.5, .0, .0, .5],
            [.0, .5, .0, .5],
        ]),
        )
        for i in range(1, n) for j in range(1, n)
    ]
    state_lens = range(1, 100)
    states = sum([state_len * 5 * [state_len % 4] for state_len in state_lens], [])
    board = Board(
        width=w,
        height=h,
        states=states,
        marchers=marchers,
    )
    os.makedirs("outputs", exist_ok=True)
    draw_board(board, os.path.join("outputs", "test.gif"), 2)
