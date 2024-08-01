import math
import cv2
import numpy as np
import copy
import serial

portstm32 = "/dev/ttyUSB0"
portdisplay = "/dev/ttyUSB1"
portstm32_2 = "/dev/ttyUSB2"
testmode = True

white_lower = np.array([200, 200, 200], dtype="uint8")
white_upper = np.array([255, 255, 255], dtype="uint8")
red_lower = np.array([0, 80, 80], dtype="uint8")
red_upper = np.array([255, 255, 255], dtype="uint8")
black_lower = np.array([0, 0, 0], dtype="uint8")
black_upper = np.array([40, 40, 40], dtype="uint8")

symbols = {0: ' ', 1: 'O', 2: 'X'}
opponent, player = 1, 2


def undistort_image(img):
    # 定义相机内参矩阵和畸变系数
    camera_matrix = np.array([[642.230498165700, 0, 581.999704543728],
                              [0, 643.146768792974, 326.934309960445],
                              [0,  0,  1]])
    distortion_coefficients = np.array(
        [-0.343363556130765, 0.124022521658576, 0, 0, -0.020903803394041])
    # 读取图像
    undistort_img = cv2.undistort(img, camera_matrix, distortion_coefficients)
    return undistort_img


def is_moves_left(board):
    for row in board:
        if 0 in row:
            return True
    return False


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


def rotationdetect():
    ret, img = cap.read()
    height, width = img.shape[:2]
    center_x = width // 2
    center_y = height // 2
    start_x = center_x - 330
    end_x = center_x + 390
    start_y = center_y - 360
    
    end_y = center_y + 360
    img = img[start_y:end_y, start_x:end_x]
    if ret:
        # 将图片从 BGR 转换为 HSV 颜色空间
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, red_lower, red_upper)
        # 将掩码应用到原始图像
        img[mask > 0] = [0, 0, 0]

        binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(binary, 120, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=3)

        cv2.imshow("img", binary)
        cv2.waitKey(1)

        binary = cv2.bitwise_not(binary)
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        largest_contour = None
        for i in range(len(contours)):
            if hierarchy[0][i][0] == -1:
                area = cv2.contourArea(contours[i])
                if area > max_area:
                    max_area = area
                    largest_contour = contours[i]

        if len(largest_contour) > 0:
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            approx = approx.reshape(-1, 2)

            approx = np.array(
                list(map(lambda p: [p[0] - img.shape[1]/2, p[1] - img.shape[0]/2], approx)))

            approx = farthest_points_in_quadrants(approx)
            if len(list(filter(lambda x: x is not None, approx.values()))) == 4:
                approx = np.array(list(approx.values()))
                approx = np.array(
                    list(map(lambda p: [p[0] + img.shape[1]/2, p[1] + img.shape[0]/2], approx)))
                return int(math.atan((approx[3][1]-approx[0][1]) /
                                     (approx[3][0]-approx[0][0]))/math.pi*180)
            else:
                return None
        else:
            return None


def get_visionboard(img):
    board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=3)
    binary = cv2.erode(binary, kernel, iterations=15)
    cv2.imshow("img", binary)
    cv2.waitKey(1)
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
        if len(list(filter(lambda x: x is not None, approx.values()))) == 4:
            approx = np.array(list(approx.values()))
            approx = np.array(
                list(map(lambda p: [p[0] + img.shape[1]/2, p[1] + img.shape[0]/2], approx)))
        else:
            return None
        size = 900
        radius = 60

        if approx.shape == (4, 2):

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
            return board
        else:
            return None
    else:
        return None
    # return board


def movechess(a, b, c, d):
    print("从", a, b, "移动棋子到", c, d, "(", (a-1)*3+(b-1), "->", (c-1)*3+(d-1), ")")
    print("命令", f"0x{(((a-1)*3+(b-1))*14+(c-1)*3+(d-1)):02X}")
    serstm32.write((((a-1)*3+(b-1))*14+(c-1)*3+(d-1)
                    ).to_bytes(1, byteorder='big'))


def compareboard(oldboard, newboard):
    ob = copy.deepcopy(oldboard)
    nb = copy.deepcopy(newboard)
    if nb == [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]
              ]:
        return -1, -1, -1, -1
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
        a = -1
        b = -1
        c = -1
        d = -1
        for i in range(len(ob)):
            for j in range(len(ob)):
                if (ob[i][j] != nb[i][j]) and nb[i][j] == 2:
                    a = i
                    b = j
                if (ob[i][j] != nb[i][j]) and ob[i][j] == 2:
                    c = i
                    d = j
        return a+1, b+1, c+1, d+1


def movechessfrom(i, a, b):
    print("从放置区", i, "移动到", a, " ", b, "(", i+9, "->", (a-1)*3+(b-1),  ")")
    print(f"0x{(((a-1)*3+(b-1))*14+i+9):02X}")
    serstm32.write((((a-1)*3+(b-1))*14+i+9).to_bytes(1, byteorder='big'))


def movechessfrom4(cmd):
    print("从放置区", int(cmd/9), "移动到", int((cmd % 9)/3)+1, " ", cmd %
          3+1, "(", int(cmd/9), "->", cmd % 9,  ")")

    print(f"0x{cmd:02X}")
    serstm32_2.write(cmd.to_bytes(1, byteorder='big'))


def rotationmode():
    angle = None
    while True:
        # img = cv2.imread("v.jpg")
        angle = rotationdetect()
        if angle != None and angle >= 0 and angle <= 90:
            # pass
            break
        else:
            continue
    angle += 45
    # 测试输出
    # angle = 0
    print("角度为", angle)
    angle = angle & 0xFF
    angle = angle | (1 << 7)
    print("命令", f"0x{angle:02X}")
    serstm32.write(angle.to_bytes(1, byteorder='big'))


def game():
    oldboard = None
    i = 0
    board = None

    # # 下棋测试
    # board = [
    #     [0, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 0]
    # ]
    # movechess(2, 2, 1, 1)

    while True:
        ret, img = cap.read()
        height, width = img.shape[:2]
        center_x = width // 2
        center_y = height // 2
        start_x = center_x - 360
        start_y = center_y - 360
        end_x = center_x + 360
        end_y = center_y + 360
        img = img[start_y:end_y, start_x:end_x]
        img[:, ::-1]
        # if ret and serdisplay.in_waiting > 0:
        # 暂时关闭串口呢
        if 1:
            # 将图片从 BGR 转换为 HSV 颜色空间
            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # 创建颜色范围的掩码
            mask = cv2.inRange(hsv_image, red_lower, red_upper)
            # 将掩码应用到原始图像
            img[mask == 0] = [0, 0, 0]
            # img = cv2.imread("v.jpg")
            # 识别棋子
            board = get_visionboard(img)

            print("获得当前数据，如下")
            if board != None:
                print_board(board)
                # 检测是否被移动，并移动棋子
                if oldboard != None:
                    a, b, c, d = compareboard(oldboard, board)
                    if a <= 0 or b <= 0 or c < 0 or d < 0:
                        print("无篡改")
                    else:
                        print("棋盘被篡改")
                        movechess(a, b, c, d)
                        board[a-1][b-1] = 0
                        board[c-1][d-1] = 2
                        print("棋子移动完成,移动后的棋盘")
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

                        # # 篡改测试
                        # board = [
                        #     [0, 0, 2],
                        #     [0, 1, 1],
                        #     [0, 0, 0]
                        # ]
                        # print("篡改后的棋盘")
                        # print_board(board)

                        if iswin(board):
                            print("游戏结束")
                            break
                else:
                    print("游戏结束")
                    break
                print("**************************")
            else:
                pass
        else:
            continue


# 0: ' ', 1: 'O', 2: 'X'
if __name__ == "__main__":

    serstm32_2 = None
    serdisplay = None

    serstm32 = serial.Serial(port=portstm32,
                             baudrate=115200,
                             parity=serial.PARITY_NONE,
                             bytesize=serial.EIGHTBITS,
                             timeout=20)
    if not serstm32.isOpen():
        print("32串口打开失败")
        exit(0)
    # 调试语句

    if testmode:
        serstm32_2 = serstm32
        serdisplay = serstm32
    else:
        serdisplay = serial.Serial(port=portdisplay,
                                   baudrate=115200,
                                   parity=serial.PARITY_NONE,
                                   stopbits=serial.STOPBITS_ONE,
                                   bytesize=serial.EIGHTBITS,
                                   timeout=1)
        if not serdisplay.isOpen():
            print("屏幕串口打开失败")
            exit(0)

        serstm32_2 = serial.Serial(port=portstm32_2,
                                   baudrate=115200,
                                   parity=serial.PARITY_NONE,
                                   bytesize=serial.EIGHTBITS,
                                   timeout=20)
        if not serstm32_2.isOpen():
            print("32串口2打开失败")
            exit(0)

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("摄像头打开失败")
        exit(0)

    while (1):
        if serdisplay.in_waiting > 0:
            cmd = serdisplay.read(1)
            print("收到指令", f"0x{cmd[0]:02X}")
            # 游戏模式
            if cmd == b'\x5A':
                print("进入游戏模式")
                rotationmode()
                game()
            # 移动一颗
            elif cmd == b'\x5B':
                # 自己写分支，填入起始位置
                print("移动黑棋0到中心", end='')
                movechessfrom(0, 2, 2)
            elif cmd == b'\x5C':
                # 自己写分支，填入起始位置
                print("移动黑棋1到中心")
                movechessfrom(1, 2, 2)
            elif cmd == b'\x5D':
                # 自己写分支，填入起始位置
                print("移动黑棋2到中心")
                movechessfrom(2, 2, 2)
            elif cmd == b'\x5E':
                # 自己写分支，填入起始位置
                print("移动黑棋3到中心")
                movechessfrom(3, 2, 2)
            elif cmd == b'\x5F':
                # 自己写分支，填入起始位置
                print("移动黑棋4到中心")
                movechessfrom(4, 2, 2)
            else:
                # 移动制定的四个棋子
                print("移动4棋子到指定位置")
                cmd = cmd[0]
                movechessfrom4(cmd)

            print("**************************")
