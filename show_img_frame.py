import cv2
import numpy as np

def play_video(frame_list,skip_frames):
    if not frame_list:
        print("No frames to display.")
        return
    frame_no = 0
    while True:
        frame = frame_list[frame_no].copy()

        cv2.putText(frame, f'Frame: {frame_no+skip_frames}', (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video', frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('d'):
            frame_no = min(frame_no + 1, len(frame_list) - 1)
        elif key == ord('a'):
            frame_no = max(frame_no - 1, 0)
        elif key == ord('s'):
            cv2.imwrite(f'data_img/frame_{frame_no}_thermalbottle.jpg', frame_list[frame_no])
            print(f'Frame {frame_no} saved.')
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()

def load_video(video_path, skip_frames,total_frames):
    cap = cv2.VideoCapture(video_path)
    #print(f"Total frames: {total_frames}")
    #total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #if skip_frames >= total_frames:
        #print("Skip frames exceed total number of frames in the video.")
        #return []
    frame_list = []

    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no >= skip_frames:
            frame_list.append(frame)
        frame_no += 1
        if len(frame_list) >= total_frames:
            break
    cap.release()
    return frame_list


# 设置绘制的参数
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.35
font_color = (255, 255, 255)
font_thickness = 1
if __name__ == "__main__":
    video_path = r'final_result/test_2.mp4'  # 替换为你的视频文件路径
    skip_frames = 500  # 跳过视频开始的帧数
    total_frames = 2000  # 总加载的帧数
    frame_list = load_video(video_path, skip_frames,total_frames)
    
    play_video(frame_list,skip_frames)
