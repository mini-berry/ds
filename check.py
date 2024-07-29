import math
import cv2
import numpy as np
import copy

white_lower = np.array([200, 200, 200], dtype="uint8")
white_upper = np.array([255, 255, 255], dtype="uint8")
red_lower = np.array([0, 0, 100], dtype="uint8")
red_upper = np.array([100, 100, 255], dtype="uint8")
black_lower = np.array([0, 0, 0], dtype="uint8")
black_upper = np.array([50, 50, 50], dtype="uint8")

symbols = {0: ' ', 1: 'O', 2: 'X'}
opponent, player = 1, 2


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


def farthest_points_in_quadrants(points):
    # 点的四个象限
    quadrants = {
        'Q2': [],  # 第二象限 (x < 0, y > 0)
        'Q3': [],  # 第三象限 (x < 0, y < 0)
        'Q4': [],   # 第四象限 (x > 0, y < 0)
        'Q1': [],  # 第一象限 (x > 0, y > 0)
    }

    for point in points:
        x, y = point
        if x >= 0 and y > 0:
            quadrants['Q1'].append(point)
        elif x < 0 and y >= 0:
            quadrants['Q2'].append(point)
        elif x <= 0 and y < 0:
            quadrants['Q3'].append(point)
        elif x > 0 and y <= 0:
            quadrants['Q4'].append(point)

    # 计算每个象限中距离原点最远的点
    farthest_points = {}
    for quad, pts in quadrants.items():
        if pts:
            distances = [np.sqrt(x**2 + y**2) for x, y in pts]
            max_index = np.argmax(distances)
            farthest_points[quad] = pts[max_index]
        else:
            farthest_points[quad] = None

    return farthest_points


def print_board(board):
    print("    1   2   3")  # 输出列号
    print("  -------------")
    i = 1
    for row in board:
        print(i, end=" ")  # 输出行号
        i += 1
        print("|", end="")
        for cell in row:
            print(f" {symbols[cell]} |", end="")
        print("\n  -------------")


def rotate_clockwise(x, y, angle):
    # 顺时针旋转
    theta = math.radians(angle)
    x_new = x * math.cos(theta) + y * math.sin(theta)
    y_new = -x * math.sin(theta) + y * math.cos(theta)
    return x_new, y_new


def get_visionboard(img):
    board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=5)
    binary = cv2.erode(binary, kernel, iterations=15)

    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        hierarchy = hierarchy[0]
        no_parent_contours = [contours[i]
                              for i in range(len(contours)) if hierarchy[i][0] == -1]
        contour = no_parent_contours[0]
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx = approx.reshape(-1, 2)
        approx = np.array(
            list(map(lambda p: [p[0] - img.shape[1]/2, p[1] - img.shape[0]/2], approx)))

        approx = farthest_points_in_quadrants(approx)
        approx = np.array(list(approx.values()))
        approx = np.array(
            list(map(lambda p: [p[0] + img.shape[1]/2, p[1] + img.shape[0]/2], approx)))

        size = 900
        radius = 60
        if len(approx) == 4:

            # 排序确定透视变换后的目标点
            rect = np.zeros((4, 2), dtype="float32")
            rect[0] = approx[1]
            rect[1] = approx[2]

            rect[2] = approx[3]
            rect[3] = approx[0]
            # 目标点，用于透视变换
            dst = np.array([
                [0, 0],
                [size - 1, 0],
                [size - 1, size - 1],
                [0, size - 1]], dtype="float32")

            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(rect, dst)
            # 进行透视变换
            img = cv2.warpPerspective(img, M, (size, size))

            for i in range(3):
                for j in range(3):
                    center_x = int((2 * j + 1) * size / 6)
                    center_y = int((2 * i + 1) * size / 6)
                    cv2.circle(img, (center_x, center_y), 60, (0, 255, 0), 2)

                    # 创建掩码
                    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
                    mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2

                    # 只处理圆形区域内的像素
                    circle_pixels = img[mask]

                    # 初始化颜色计数
                    color_count = {'1': 0, '0': 0, '2': 0}

                    # 计算颜色区域
                    white_mask = np.all((white_lower <= circle_pixels) & (
                        circle_pixels <= white_upper), axis=1)
                    red_mask = np.all((red_lower <= circle_pixels) & (
                        circle_pixels <= red_upper), axis=1)
                    black_mask = np.all((black_lower <= circle_pixels) & (
                        circle_pixels <= black_upper), axis=1)

                    color_count['1'] = np.sum(white_mask)
                    color_count['0'] = np.sum(red_mask)
                    color_count['2'] = np.sum(black_mask)

                    # 找出最主要的颜色
                    dominant_color = max(color_count, key=color_count.get)

                    board[i][j] = int(dominant_color)
            cv2.imwrite("ans.jpg", img)
        return board
    else:
        return None
    # return board


def movechess(x, y, a, b):
    print("从" + str(x)+" "+str(y)+"移动棋子到" + str(a)+" "+str(b))
    print("命令", (x)*3+y, (a)*3+b)


def compareboard(board1, board2):
    ob = copy.deepcopy(board1)
    nb = copy.deepcopy(board2)
    for i in range(len(nb)):
        for j in range(len(nb[i])):
            if nb[i][j] == 1:
                nb[i][j] = 0
    for i in range(len(ob)):
        for j in range(len(ob[i])):
            if ob[i][j] == 1:
                ob[i][j] = 0

    if ob == nb:
        return -1, -1, -1, -1
    else:
        print("移动前的棋盘")
        a = -1
        b = -1
        c = -1
        d = -1
        print(a, b, c, d)
        for i in range(len(ob)):
            for j in range(len(ob)):
                if (ob[i][j] != nb[i][j]) and nb[i][j] == 2:
                    a = i
                    b = j
                if (ob[i][j] != nb[i][j]) and ob[i][j] == 2:
                    print(i, j)
                    c = i
                    d = j
        print(a, b, c, d)
        return a+1, b+1, c+1, d+1


def movechessfrom(i, a, b):
    print("从放置区"+str(i)+"移动到"+str(a)+" "+str(b))
    print("命令", i+9, (a-1)*3+b-1)


# 0: ' ', 1: 'O', 2: 'X'
if __name__ == "__main__":
    oldboard = None
    i = 0
    board = None
    while True:
        img = cv2.imread("v.jpg")
        # 识别棋子
        board = get_visionboard(img)
        print("获得当前数据，如下")
        board = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        # 检测是否被移动，并移动棋子
        if oldboard != None:
            a, b, c, d = compareboard(oldboard, board)
            if a == -1:
                print("棋盘相同")
            else:
                print("棋盘不同")
                movechess(a, b, c, d)
                board[a-1][b-1] = 0
                board[c-1][d-1] = 2
                print("棋子移动完成,移动后的棋盘")
            # 读取图像

        if board != None:
            print_board(board)
        if not iswin(board):
            # 找到玩家1（黑棋）的最佳落子点
            best_move = find_best_move(board)
            if best_move == (-1, -1):
                print("没有可用的落子点。")
                break
            else:
                print(
                    f"玩家1（黑棋）的最佳落子点是：第 {1+best_move[0]} 行，第 {1+best_move[1]} 列，落子后")
                movechessfrom(i, best_move[0]+1, best_move[1]+1)
                i += 1
                # 执行落子
                board[best_move[0]][best_move[1]] = player
                oldboard = board
                print_board(board)
                if iswin(board):
                    print("游戏结束")
                    break
        else:
            print("游戏结束")
            break
