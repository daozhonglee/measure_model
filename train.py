import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
# 试验记录

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8s.yaml')
    model.load(r'runs\train-data-add-1280-s\exp\weights\best.pt') # loading pretrain weights
    model.train(data='ultralytics\cfg\datasets\mydataset.yaml',  
                cache=False,
                imgsz=1280,
                epochs=10,
                batch=4,
                close_mosaic=0,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train-data-add-1280-s',
                name='exp',
                )