import numpy as np

def check_winner(board):
    for i in range(3):
        if board[i, 0] == board[i, 1] == board[i, 2] != 0:
            return board[i, 0]
        if board[0, i] == board[1, i] == board[2, i] != 0:
            return board[0, i]
    if board[0, 0] == board[1, 1] == board[2, 2] != 0:
        return board[0, 0]
    if board[0, 2] == board[1, 1] == board[2, 0] != 0:
        return board[0, 2]
    return 0

def is_moves_left(board):
    return np.any(board == 0)

def minimax(board, depth, is_max):
    score = check_winner(board)
    if score == 2:
        return 10 - depth
    if score == 1:
        return depth - 10
    if not is_moves_left(board):
        return 0

    if is_max:
        best = -1000
        for i in range(3):
            for j in range(3):
                if board[i, j] == 0:
                    board[i, j] = 2
                    best = max(best, minimax(board, depth + 1, not is_max))
                    board[i, j] = 0
        return best
    else:
        best = 1000
        for i in range(3):
            for j in range(3):
                if board[i, j] == 0:
                    board[i, j] = 1
                    best = min(best, minimax(board, depth + 1, not is_max))
                    board[i, j] = 0
        return best

def find_best_move(board):
    best_val = -1000
    best_move = (-1, -1)

    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                board[i, j] = 2
                move_val = minimax(board, 0, False)
                board[i, j] = 0
                if move_val > best_val:
                    best_move = (i, j)
                    best_val = move_val

    return best_move

# 示例棋盘，0代表空，1代表白棋，2代表黑棋
board = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]])

# 找到黑棋（2）的最优解
best_move = find_best_move(board)
print("The best move for black (2) is:", best_move)
