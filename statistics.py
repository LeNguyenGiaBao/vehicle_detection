import time
import cv2
import os
from get_data import get_image
from model import SSD, Predictor

def count_number_per_class(id_camera, labels):
    class_names = ['BACKGROUND', 'motorcycle', 'car', 'bus', 'truck']
    data = {
        "id": id_camera,
        "motorcycle": 0,
        "car": 0,
        "bus": 0,
        "truck": 0
    }

    for c in labels:
        cls = class_names[int(c)]
        data[cls] += 1
    
    return data

def save_predict_image(img, boxes, labels, probs, current_time):
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 1)
        label = f"{class_names[labels[i]][0]}: {probs[i]:.2f}"
        cv2.putText(img, label,
                    (int(box[0]) + 10, int(box[1]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    (0, 0, 0),
                    2) # line type

    cv2.imwrite(data_path + '/' + current_time + '.jpg', img)

class_names = ['BACKGROUND', 'motorcycle', 'car', 'bus', 'truck']
model_path = './models/vgg16-ssd-Epoch-170-Loss-1.8997838258743287.pth'
net = SSD(len(class_names), is_test=True)
net.load(model_path)
predictor = Predictor(net, nms_method='soft', candidate_size=200)
id_camera = '5d8cd542766c880017188948'
current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
data_path = './data/' + id_camera + '_' + current_time
data_file = open(data_path + '.csv', 'w')
os.mkdir(data_path)

while True:
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    print(current_time)
    img = get_image(id_camera)
    boxes, labels, probs = predictor.predict(img, 50, 0.3)
    data = count_number_per_class(id_camera, labels)
    save_predict_image(img, boxes, labels, probs, current_time)
    data_file.write(current_time + ',' + ','.join(map(str, data.values())) + '\n')
    time.sleep(15)

data_file.close()

