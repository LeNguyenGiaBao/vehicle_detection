import time
import cv2
import os
from utils.utils import get_image, get_cur_time, draw_boxes, count_number_per_class
from model import SSD, Predictor


class_names = ['BACKGROUND', 'motorcycle', 'car', 'bus', 'truck']
model_path = './models/vgg16-ssd-Epoch-170-Loss-1.8997838258743287.pth'
net = SSD(len(class_names), is_test=True)
net.load(model_path)
predictor = Predictor(net, nms_method='soft', candidate_size=200)
id_camera = '5d8cd542766c880017188948'
current_time = get_cur_time()
data_path = './data/' + id_camera + '_' + current_time
data_file = open(data_path + '.csv', 'w')
os.mkdir(data_path)

while True:
    current_time = get_cur_time()
    print(current_time)
    img = get_image(id_camera)
    boxes, labels, probs = predictor.predict(img, 50, 0.3)
    img = draw_boxes(img, boxes, labels, probs, class_names)
    data = count_number_per_class(class_names, labels)
    cv2.imwrite(data_path + '/' + current_time + '.jpg', img)
    data_file.write(current_time + ',' + id_camera + ',' + ','.join(map(str, data.values())) + '\n')
    time.sleep(15)
