from yoloface.face_detector import YoloDetector
import numpy as np
from PIL import Image

model = YoloDetector(target_size=720, device="cuda:0", min_face=90)
orgimg = np.array(Image.open('./data/helen/trainset/1629243_1.jpg'))
bboxes,points = model.predict(orgimg)
print(bboxes[0])
print(points)