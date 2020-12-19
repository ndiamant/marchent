import os
import argparse
import numpy as np
from typing import Callable, List
from functools import partial

from marchent.marcher import (
    Marcher,
    # color rules
    rand_color,
    next_hsv,
    lighten,
    turn_color,
    # move rules
    ortho_split,
    angle_split,
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


def tree() -> Board:
    h = 500
    w = 1000
    marcher_pairs = [
        [
            Marcher(
                i, j,
                color=np.array([90, 90, 30]).astype(np.uint8),
                state_transitions=np.array([
                    [1, 0, 0],  # run, split, stop
                    [.5, .5, 0],  # run, split, stop
                ]),
                move_transitions=np.array([
                    [.0, .03, 0, .97],  # right, left, down, up
                    [0, .2, 0, .8],  # right, left, down, up
                ]),
                split_color=partial(turn_color, next_colors=[
                    np.array([10, np.random.randint(128), 10]).astype(np.uint8)
                    for _ in range(5)
                ]),
                split_move_transitions=partial(angle_split, theta=np.pi / 20),
            ),
            Marcher(
                i + 1, j,
                color=np.array([90, 90, 30]).astype(np.uint8),
                state_transitions=np.array([
                    [1, .0, 0],  # run, split, stop
                    [.5, .5, 0],  # run, split, stop
                ]),
                move_transitions=np.array([
                    [.03, .0, 0, .97],  # right, left, down, up
                    [.2, 0, 0, .8],  # right, left, down, up
                ]),
                split_color=partial(turn_color, next_colors=[
                    np.array([10, np.random.randint(128), 10]).astype(np.uint8)
                    for _ in range(5)
                ]),
                split_move_transitions=partial(angle_split, theta=np.pi / 20),
                ),
            ]
        for i in range(0, w, 30)
        for j in [h - 1, h // 2]
    ]
    marchers = sum(marcher_pairs, [])
    states = [0] * 75 + [1] * 500
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
    tree,
]


if __name__ == '__main__':
    os.makedirs("outputs", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=[f.__name__ for f in EXAMPLES])
    f_name = parser.parse_args().mode
    board = next(f for f in EXAMPLES if f.__name__ == f_name)()
    draw_board(board, os.path.join("outputs", f"{f_name}.gif"), 1)
