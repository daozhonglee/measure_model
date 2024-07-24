import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO(r'/Users/shanquan/code/opencv_code/measure/v2/upload_files/runs/train-data-add-3-1280-s/exp3/weights/best.onnx') # select your model.pt path
    model.predict(source='final_video/2.MOV',
                project='runs/detect',
                name='exp',
                save=True,
                save_txt=False,
                 visualize=True # visualize model features maps
                )