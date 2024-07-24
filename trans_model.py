from ultralytics import YOLO


# 调用入口尽可能的简单，只需要传入模型路径即可
model = YOLO(r"runs\train-data-add-3-1280-s\exp3\weights\best.pt")
# 保存在同样的路径
model.export(format = "onnx")  # export the model to onnx format