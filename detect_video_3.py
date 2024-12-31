import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO
import cv2
import os
import collections
import numpy as np

# debug mode   1:normal   2:all
debug_mode = 1
# 支持库：onnxruntime-gpu
model = YOLO(r'/Users/shanquan/code/opencv_code/measure/v2/upload_files/runs/train-data-add-3-1280-s/exp3/weights/best.onnx')  # pretrained YOLOv8n model

# Initialize video capture
num_video = 2
video_path = rf'/Users/shanquan/Movies/basketball/b1.mp4'
resize_flag = False

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Save path for video
save_path_video = r'final_new_result'
if not os.path.exists(save_path_video):
    os.makedirs(save_path_video)

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(os.path.join(save_path_video, f'{num_video}.mp4'), fourcc, fps, size)

# Score counting and flag status
score_count = 0
shot_count = 0
last_shot_frame = -30
flag1, flag2 = False, False


def find_largest_basketboard(boxes, labels):
    max_area = 0
    largest_basketboard_box = None
    for box, label in zip(boxes, labels):
        if label == 1:  # Assuming label 1 is Basketboard
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > max_area:
                max_area = area
                largest_basketboard_box = box
    return largest_basketboard_box


def find_nearest_ball_and_basket(boxes, labels, basketboard_box):
    nearest_ball_box = None
    basket_box_inside = None
    min_distance = float('inf')

    for box, label in zip(boxes, labels):
        if label == 2:  # Assuming label 2 is Ball
            ball_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            basketboard_center = (
            (basketboard_box[0] + basketboard_box[2]) / 2, (basketboard_box[1] + basketboard_box[3]) / 2)
            distance = ((ball_center[0] - basketboard_center[0]) ** 2 + (
                        ball_center[1] - basketboard_center[1]) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_ball_box = box
        elif label == 0 and len(basketboard_box) > 0:  # Assuming label 0 is Basket
            if basketboard_box[0] <= box[0] <= basketboard_box[2] and basketboard_box[0] <= box[2] <= basketboard_box[
                2]:
                basket_box_inside = box

    return nearest_ball_box, basket_box_inside


def check_goal(ball_box, basket_box, flag1, flag2, extended_pixels=50):
    extended_pixels = (basket_box[3] - basket_box[1]) * 2
    # Calculate midpoints and boundaries
    # xyxy 球下边界的中点
    ball_mid_bottom = (ball_box[0] + ball_box[2]) / 2, ball_box[3]
    # 球上边界的重点
    ball_mid_top = (ball_box[0] + ball_box[2]) / 2, ball_box[1]
    # 球的中心

    ball_center = (ball_box[0] + ball_box[2]) / 2, (ball_box[1] + ball_box[3]) / 2
    # 不优雅的写法
    ball_mid_top = ball_center
    # 篮网的上边界
    basket_top = basket_box[1]
    basket_buttom = basket_box[3]
    # 篮网的下边界，扩展extended_pixels像素
    basket_bottom_extended = basket_box[3] + extended_pixels
    # 上边界扩展

    basket_top_extended = basket_box[1] - extended_pixels
    # 篮网的左边界
    basket_left = basket_box[0]
    # 篮网的右边界
    basket_right = basket_box[2]
    # 是篮网的高度的2倍

    # Visualize extended basket boundary
    # rectangle 绘制参数介绍：左上角坐标，右下角坐标，颜色，线宽
    if debug_mode == 2:
        cv2.rectangle(frame,
                      (int(basket_box[0]), int(basket_box[3])),
                      (int(basket_box[2]), int(basket_box[3] + extended_pixels)),
                      (0, 255, 0), 2)
        # 绘制上边界的扩展
        cv2.rectangle(frame,
                      (int(basket_box[0]), int(basket_box[1] - extended_pixels)),
                      (int(basket_box[2]), int(basket_box[1])),
                      (255, 0, 0), 2)

    # 进行进球的逻辑判断
    # Check conditions including x-axis alignment

    if basket_left <= ball_mid_bottom[0] <= basket_right and basket_top_extended <= ball_mid_bottom[1] <= basket_buttom:
        flag1 = True

    # 绘制ball_mid_top
    cv2.circle(frame, (int(ball_mid_top[0]), int(ball_mid_top[1])), 2, (0, 255, 0), -1)
    if flag1 and basket_left <= ball_mid_top[0] <= basket_right and basket_buttom <= ball_mid_top[
        1] <= basket_bottom_extended:
        flag2 = True
    if debug_mode == 2:
        cv2.putText(frame,
                    f'flag1_x: {int(basket_left)}, {int(ball_mid_bottom[0])},{int(basket_right)}',
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
        cv2.putText(frame,
                    f'flag1_y: {int(basket_top_extended)}, {int(ball_mid_bottom[1])},{int(basket_top)}',
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
        cv2.putText(frame,
                    f'flag2_x: {int(basket_left)}, {int(ball_mid_top[0])},{int(basket_right)}',
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
        cv2.putText(frame,
                    f'flag2_y: {int(basket_buttom)}, {int(ball_mid_top[1])},{int(basket_bottom_extended)}',
                    (10, 180), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)

    return flag1, flag2


def check_shot_attempt(ball_box, basketboard_box, frame_count, last_shot_frame):
    # Check if the ball is within the basketboard boundaries
    if (basketboard_box[0] <= ball_box[0] <= basketboard_box[2] and
            basketboard_box[1] <= ball_box[1] <= basketboard_box[3] and
            basketboard_box[0] <= ball_box[2] <= basketboard_box[2] and
            basketboard_box[1] <= ball_box[3] <= basketboard_box[3]):
        if frame_count > last_shot_frame + 100:  # Check if 30 frames have passed since the last shot
            return True
    return False


def draw_trajectory(frame, positions, max_frames=10, max_thickness=5):
    # Ensure the thickness decreases as the points get older
    reversed_list = list(reversed(positions))
    if len(reversed_list) > 1:
        for i in range(1, len(reversed_list)):
            thickness = max_thickness - int((max_thickness - 1) * (i / max_frames))
            if thickness < 1:
                thickness = 1
            if debug_mode == 2:
                cv2.line(frame, reversed_list[i - 1], reversed_list[i], (0, 255, 0), thickness)


if __name__ == '__main__':
    # Process video frames
    frame_count = 0
    flag_pause = 0
    flag1_patience, flag2_patience = 0, 0
    flag1_flag = True
    flag2_flag = True
    ball_positions = collections.deque(maxlen=10)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # 首先对图像进行预处理，找到较短的边，然后用灰色填充为一个矩形图像
            # 获取图像的宽度和高度

            if resize_flag == True:
                height, width = frame.shape[:2]
                size = max(width, height)
                new_image = np.zeros((size, size, 3), np.uint8)
                # 计算填充边界
                top = (size - height) // 2
                bottom = size - height - top
                left = (size - width) // 2
                right = size - width - left

                # 填充并居中图像
                frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

                cv2.imshow('square_image', frame)
                continue
            # Predict objects in the frame
            result = model.predict(frame, imgsz=1280)[0]

            ball_box = None
            basket_box = None
            basketboard_box = None
            nearest_ball_box = None
            basket_box_inside = None
            # Extract bounding boxes
            # for result in results:

            boxes = result.boxes.xyxy
            labels = result.boxes.cls
            for i, box in enumerate(boxes):
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                cv2.putText(frame, f'{labels[i]}', (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 0, 0), 2)
            # Find the largest basketboard
            largest_basketboard_box = find_largest_basketboard(boxes, labels)
            # print(largest_basketboard_box)
            if largest_basketboard_box is not None:
                if len(largest_basketboard_box) > 0:
                    # Find the nearest ball and the basket inside the largest basketboard
                    nearest_ball_box, basket_box_inside = find_nearest_ball_and_basket(boxes, labels,
                                                                                       largest_basketboard_box)
                    # 找到这三个之后，绘制

                cv2.rectangle(frame, (int(largest_basketboard_box[0]), int(largest_basketboard_box[1])),
                              (int(largest_basketboard_box[2]), int(largest_basketboard_box[3])), (0, 255, 0), 2)
                cv2.putText(frame, f'Basketboard',
                            (int(largest_basketboard_box[0]), int(largest_basketboard_box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if nearest_ball_box is None:
                    ball_positions = collections.deque(maxlen=10)

                if nearest_ball_box is not None:

                    # Draw the trajectory of the ball
                    ball_center = (int((nearest_ball_box[0] + nearest_ball_box[2]) / 2),
                                   int((nearest_ball_box[1] + nearest_ball_box[3]) / 2))
                    ball_positions.append(ball_center)
                    # 把ball_positions列表倒过来

                    # ball_positions = list(reversed(ball_positions))
                    draw_trajectory(frame, ball_positions, max_frames=10, max_thickness=5)
                    # cv2.putText(frame,{str(len(ball_positions)),})
                    if debug_mode == 2:
                        cv2.rectangle(frame, (int(nearest_ball_box[0]), int(nearest_ball_box[1])),
                                      (int(nearest_ball_box[2]), int(nearest_ball_box[3])), (0, 0, 255), 2)
                        cv2.putText(frame, f'Ball', (int(nearest_ball_box[0]), int(nearest_ball_box[1] - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if basket_box_inside is not None:
                    cv2.rectangle(frame, (int(basket_box_inside[0]), int(basket_box_inside[1])),
                                  (int(basket_box_inside[2]), int(basket_box_inside[3])), (255, 255, 0), 2)
                    cv2.putText(frame, f'Basket', (int(basket_box_inside[0]), int(basket_box_inside[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            ball_box = nearest_ball_box
            basket_box = basket_box_inside
            basketboard_box = largest_basketboard_box

            # Check for goal and visualize
            if ball_box is not None and basket_box is not None:
                flag1, flag2 = check_goal(ball_box, basket_box, flag1, flag2)

            if ball_box is not None and basketboard_box is not None:
                if check_shot_attempt(ball_box, basketboard_box, frame_count, last_shot_frame):
                    shot_count += 1
                    last_shot_frame = frame_count  # Update the last shot frame count
            # flag1和flag2的留存帧数
            if flag1 and flag1_flag:
                flag1_patience = frame_count + 120
                flag1_flag = False
            if flag2 and flag2_flag:
                flag2_patience = frame_count + 120
                flag2_flag = False

            if flag1_patience < frame_count:
                flag1 = False
                flag1_flag = True
            if flag2_patience < frame_count:
                flag2 = False
                flag2_flag = True

            # Draw flags and score
            if debug_mode == 2:
                cv2.putText(frame, f'Flags: {flag1}, {flag2}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # 进球次数
            cv2.putText(frame, f'Score: {score_count}', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
            # 发球次数
            cv2.putText(frame, f'Shot_: {shot_count}', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
            # 当前第几帧
            cv2.putText(frame, f'Frame: {frame_count}', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (140, 0, 0), 2)
            # 计算命中率

            if shot_count > 0:
                hit_rate = (score_count / shot_count) * 100
                cv2.putText(frame, f'Hit rate: {hit_rate:.2f}%', (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

            if flag1 and flag2 and flag_pause < frame_count:
                score_count += 1
                flag1, flag2 = False, False  # Reset flags after scoring
                flag1_flag, flag2_flag = True, True
                flag_pause = frame_count + 60

            cv2.imshow('frame', frame)
            # quit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if shot_count > 51:
                break
            video_writer.write(frame)
            frame_count += 1

            # if frame_count > 2000:
            # break
        else:
            break

    # Cleanup
    cap.release()
    video_writer.release()
