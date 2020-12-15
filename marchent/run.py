import os
import argparse
import numpy as np
from typing import Callable, List

from marchent.marcher import (
    Marcher,
    MarcherState,
)
from marchent.board import (
    draw_board,
    Board,
    ortho_marcher,
    perturbed_ortho_marcher,
    copy_darker,
    entropy_copy,
)


# Examples
def ortho_split() -> Board:
    h = 1000
    w = 1000
    marchers = [Marcher(
        w // 2, h // 2,
        color=255,
        state_transitions=np.array([
            [.9, .1, 0],  # run, split, stop
        ]),
        move_transitions=np.array([
            [1, 0, 0, 0],  # right, left, down, up
        ]),
    )
    ]
    states = [0] * 100000
    return Board(
        width=w,
        height=h,
        states=states,
        marchers=marchers,
        split_marcher=ortho_marcher,
    )


def diag_ortho_split() -> Board:
    h = 1000
    w = 1000
    marchers = [Marcher(
        w // 2, h // 2,
        color=255,
        state_transitions=np.array([
            [.9, .1, 0],  # run, split, stop
        ]),
        move_transitions=np.array([
            [.5, 0, .5, 0],
        ]),
        )
    ]
    states = [0] * 100000
    return Board(
        width=w,
        height=h,
        states=states,
        marchers=marchers,
        split_marcher=ortho_marcher,
    )


def perturb_ortho_split() -> Board:
    h = 1000
    w = 1000
    marchers = [Marcher(
        w // 3, 0,
        color=255,
        state_transitions=np.array([
            [.99, .01, 0],  # run, split, stop
            [.5, .5, 0],  # run, split, stop
        ]),
        move_transitions=np.array([
            [.1, 0, .9, 0],  # right, left, down, up
            [0, .1, .9, 0],
        ]),
        )
    ]
    states = ([0] * 10 + [1] * 2) * 100
    return Board(
        width=w,
        height=h,
        states=states,
        marchers=marchers,
        split_marcher=perturbed_ortho_marcher,
    )


def zig_zag() -> Board:
    h = 600
    w = 1000
    marchers = [Marcher(
        np.random.randint(w), np.random.randint(h // 10),
        color=255,
        state_transitions=np.array([
            [.9, .1, .0],  # run, split, stop
            [.9, .1, .0],  # run, split, stop
        ]),
        move_transitions=np.array([
            [.2, 0, .8, 0],  # right, left, down, up
            [0, .2, .8, 0],  # right, left, down, up
        ]),
        )
        for _ in range(w // 2)
    ]
    states = 100 * ([0] * 50 + [1] * 50)
    return Board(
        width=w,
        height=h,
        states=states,
        marchers=marchers,
        split_marcher=copy_darker,
    )


def entropy() -> Board:
    h = 500
    w = 1000
    marchers = [
        Marcher(
            0, h // 2,
            color=255,
            state_transitions=np.array([
                [.1, .9, .0],  # run, split, stop
            ]),
            move_transitions=np.array([
                [1, 0, 0, 0],  # right, left, down, up
            ])
        ),
    ]
    states = [0] * 1000
    return Board(
        width=w,
        height=h,
        states=states,
        marchers=marchers,
        split_marcher=entropy_copy,
    )


EXAMPLES: List[Callable[[], Board]] = [
    ortho_split,
    diag_ortho_split,
    perturb_ortho_split,
    zig_zag,
    entropy,
]


if __name__ == '__main__':
    os.makedirs("outputs", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=[f.__name__ for f in EXAMPLES])
    f_name = parser.parse_args().mode
    board = next(f for f in EXAMPLES if f.__name__ == f_name)()
    draw_board(board, os.path.join("outputs", f"{f_name}.gif"), 1)
