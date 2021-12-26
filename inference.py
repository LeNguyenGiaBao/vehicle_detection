import cv2
import argparse
import os
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
for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.putText(img, label,
                (int(box[0]) + 20, int(box[1]) + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
cv2.imwrite('./data/result.jpg', img)
print('Save result at: ', os.path.abspath('./data/result.jpg'))
