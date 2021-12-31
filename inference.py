import cv2
import argparse
import os
from model import SSD, Predictor
from utils.utils import draw_boxes


class_names = ['BACKGROUND', 'motorcycle', 'car', 'bus', 'truck']
model_path = './models/vgg16-ssd-Epoch-170-Loss-1.8997838258743287.pth'
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Input iamge')
args = parser.parse_args()
image_path = args.input
net = SSD(len(class_names), is_test=True)
net.load(model_path)
predictor = Predictor(net, nms_method='soft', candidate_size=200)
img = cv2.imread(image_path)
boxes, labels, probs = predictor.predict(img, 10, 0.3)
img = draw_boxes(img, boxes, labels, probs, class_names)
cv2.imwrite('./data/result.jpg', img)
print('Save result at: ', os.path.abspath('./data/result.jpg'))
