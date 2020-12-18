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
    unbalanced_ortho_marcher,
    perturbed_ortho_marcher,
    unbalanced_ortho_marcher_faster_spawn,
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


def ortho_split_small() -> Board:
    h = 250
    w = 750
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


def square_spiral() -> Board:
    h = 1000
    w = 1000
    marchers = [Marcher(
        0, 0,
        color=255,
        state_transitions=np.array([
            [.9, .1, 0],  # run, split, stop
        ]),
        move_transitions=np.array([
            [1, 0, 0, 0],
        ]),
        )
    ]
    states = [0] * 100000
    return Board(
        width=w,
        height=h,
        states=states,
        marchers=marchers,
        split_marcher=unbalanced_ortho_marcher,
    )


def square_spiral_2() -> Board:
    h = 500
    w = 500
    marchers = [
        Marcher(
            0, 0,
            color=255,
            state_transitions=np.array([
                [.98, .02, 0],  # run, split, stop
            ]),
            move_transitions=np.array([
                [.9, 0, .1, 0]]
            ),
        ),
        Marcher(
            w - 1, h - 1,
            color=255,
            state_transitions=np.array([
                [.98, .02, 0],  # run, split, stop
            ]),
            move_transitions=np.array([
                [0, .9, 0, .1]]
            ),
        )
    ]
    states = [0] * 100000
    return Board(
        width=w,
        height=h,
        states=states,
        marchers=marchers,
        split_marcher=unbalanced_ortho_marcher_faster_spawn,
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


def clash() -> Board:
    h = 250
    w = 750
    n = 70
    marchers = [Marcher(
        np.random.randint(w // 3), np.random.randint(h),
        color=np.random.randint(255),
        state_transitions=np.array([
            [.9, .1, 0],  # run, split, stop
        ]),
        move_transitions=np.array([
            [.8, 0, .1, .1],  # right, left, down, up
        ]),
        )
        for _ in range(n)
    ]
    marchers += [
        Marcher(
            np.random.randint(2 * w // 3, w), np.random.randint(h),
            color=np.random.randint(255),
            state_transitions=np.array([
                [.9, .1, 0],  # run, split, stop
            ]),
            move_transitions=np.array([
                [0, .8, .1, .1],  # right, left, down, up
            ]),
        )
        for _ in range(n)
    ]
    states = ([0] * 1000) * 100
    return Board(
        width=w,
        height=h,
        states=states,
        marchers=marchers,
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
    ortho_split_small,
    diag_ortho_split,
    perturb_ortho_split,
    zig_zag,
    entropy,
    square_spiral,
    square_spiral_2,
    clash,
]


if __name__ == '__main__':
    os.makedirs("outputs", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=[f.__name__ for f in EXAMPLES])
    f_name = parser.parse_args().mode
    board = next(f for f in EXAMPLES if f.__name__ == f_name)()
    draw_board(board, os.path.join("outputs", f"{f_name}.gif"), 1)
