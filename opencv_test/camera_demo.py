import cv2

if __name__ == '__main__':
    # 初始化摄像头
    cap = cv2.VideoCapture(0)  # 参数0通常表示默认的摄像头

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()

    # 循环读取摄像头的每一帧
    while True:
        # 读取一帧
        ret, frame = cap.read()

        # 如果正确读取帧，ret为True
        if not ret:
            print("Error: Could not read frame")
            break

        # 显示帧
        cv2.imshow('Camera Capture', frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头
    cap.release()
    # 销毁所有窗口
    cv2.destroyAllWindows()