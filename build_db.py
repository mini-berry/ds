import math

# Define the player and opponent
player, opponent = 2, 1

# Check if there are moves remaining on the board


def print_board(board):
    symbols = {0: ' ', 1: 'O', 2: 'X'}
    print("y -------------")
    i = 3
    for row in board:
        print(i, end=" ")  # 输出行号
        i -= 1
        print("|", end="")
        for cell in row:
            print(f" {symbols[cell]} |", end="")
        print("\n  -------------")
    print("x   1   2   3")  # 输出列号


def is_moves_left(board):
    for row in board:
        if 0 in row:
            return True
    return False

# Evaluate the board


def iswin(board):
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] != 0:
            return True

    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != 0:
            return True

    if board[0][0] == board[1][1] == board[2][2] != 0:
        return True

    if board[0][2] == board[1][1] == board[2][0] != 0:
        return True


def evaluate(board):
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2]:
            if board[row][0] == player:
                return 10
            elif board[row][0] == opponent:
                return -10

    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col]:
            if board[0][col] == player:
                return 10
            elif board[0][col] == opponent:
                return -10

    if board[0][0] == board[1][1] == board[2][2]:
        if board[0][0] == player:
            return 10
        elif board[0][0] == opponent:
            return -10

    if board[0][2] == board[1][1] == board[2][0]:
        if board[0][2] == player:
            return 10
        elif board[0][2] == opponent:
            return -10

    return 0

# Minimax function


def minimax(board, depth, is_max):
    score = evaluate(board)

    if score == 10:
        return score - depth
    if score == -10:
        return score + depth
    if not is_moves_left(board):
        return 0

    if is_max:
        best = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    board[i][j] = player
                    best = max(best, minimax(board, depth + 1, not is_max))
                    board[i][j] = 0
        return best
    else:
        best = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    board[i][j] = opponent
                    best = min(best, minimax(board, depth + 1, not is_max))
                    board[i][j] = 0
        return best

# Find the best move


def find_best_move(board):
    best_val = -math.inf
    best_move = (-1, -1)
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                board[i][j] = player
                move_val = minimax(board, 0, False)
                board[i][j] = 0
                if move_val > best_val:
                    best_move = (i, j)
                    best_val = move_val
    return best_move

# 下左上角时，仅下中间平局，其余均赢
board = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
]

while (1):
    print_board(board)
    best_move = find_best_move(board)
    if best_move == (-1, -1):
        print("Game over")
        break
    print(f"The best move is at position: {best_move}")
    board[best_move[0]][best_move[1]] = player
    player, opponent = opponent, player
    if (iswin(board)):
        print_board(board)
        print("Game over")
        break
