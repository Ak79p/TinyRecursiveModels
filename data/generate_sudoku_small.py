#!/usr/bin/env python3
"""
Small Sudoku dataset generator.

Produces data/sudoku_small.jsonl with one JSON object per line:
{
  "id": int,
  "input": [[...9 rows...]],
  "target": [[...9 rows...]]
}

This generator:
- creates solved Sudoku boards by backtracking,
- removes `remove_count` cells randomly to make puzzles,
- produces `num_puzzles` puzzles.

Designed for small local experiments.
"""
import random, json
from copy import deepcopy

# Backtracking solver/generator adapted for 9x9 Sudoku
def valid(board, r, c, val):
    # row and col
    for k in range(9):
        if board[r][k] == val or board[k][c] == val:
            return False
    # 3x3 box
    br, bc = 3 * (r // 3), 3 * (c // 3)
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            if board[i][j] == val:
                return False
    return True

def solve(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                for val in range(1, 10):
                    if valid(board, i, j, val):
                        board[i][j] = val
                        if solve(board):
                            return True
                        board[i][j] = 0
                return False
    return True

def generate_solved_board():
    board = [[0]*9 for _ in range(9)]
    # Fill diagonal 3x3 blocks with random permutations to help generation speed
    for block in range(3):
        nums = list(range(1,10))
        random.shuffle(nums)
        br = block * 3
        for i in range(3):
            for j in range(3):
                board[br + i][br + j] = nums.pop()
    # Solve to fill the whole board
    solve(board)
    return board

def make_puzzle(solved_board, remove_count=40):
    puzzle = deepcopy(solved_board)
    cells = [(i,j) for i in range(9) for j in range(9)]
    random.shuffle(cells)
    removed = 0
    for (r,c) in cells:
        if removed >= remove_count:
            break
        # remove and test solvability (quick check: try solving the board)
        backup = puzzle[r][c]
        puzzle[r][c] = 0
        # make a copy and try to solve; if solvable, keep removal, else revert
        temp = deepcopy(puzzle)
        if solve(temp):
            removed += 1
        else:
            puzzle[r][c] = backup
    return puzzle

def main(num_puzzles=100, remove_count=40, outpath="data/sudoku_small.jsonl"):
    random.seed(42)
    written = 0
    with open(outpath, "w") as f:
        for idx in range(num_puzzles):
            solved = generate_solved_board()
            puzzle = make_puzzle(solved, remove_count=remove_count)
            obj = {"id": idx, "input": puzzle, "target": solved}
            f.write(json.dumps(obj) + "\n")
            written += 1
            if (idx+1) % 10 == 0:
                print(f"Generated {idx+1} puzzles")
    print(f"Done. Wrote {written} puzzles to {outpath}")

if __name__ == "__main__":
    # small default: 100 puzzles, remove 40 cells => medium difficulty
    main(num_puzzles=100, remove_count=40, outpath="data/sudoku_small.jsonl")