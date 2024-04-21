import cv2
from yolov7 import YOLOv7
from PIL import Image
import numpy as np

model_path =  "models/yolov7.onnx"

detector = YOLOv7(model_path, conf_thres=0.1, iou_thres=0.2, official_nms=True)

image = Image.open('inference/image.png')
image = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)


prepared_image = detector.prepare_input(image)

for i in range(20):
    inference = detector.inference(prepared_image)