import cv2 
from tornado import httpclient
from PIL import Image
import numpy as np 
import io 
import time 
http_client = httpclient.HTTPClient()

def get_image(id_camera):
    response = http_client.fetch("http://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id={}".format(id_camera))
    image = Image.open(io.BytesIO(response.body))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def draw_boxes(img, boxes, labels, probs, class_names):
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 1)
        label = f"{class_names[labels[i]][0]}: {probs[i]:.2f}"
        cv2.putText(img, label,
                    (int(box[0]) + 10, int(box[1]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    (127, 0, 127),
                    2) # line type

    return img

def get_cur_time():
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

def count_number_per_class(class_names, labels):
    class_names = class_names[1:]
    data = dict.fromkeys(class_names, 0)

    for c in labels:
        cls = class_names[int(c)-1]
        if cls not in data:
            data[cls] = 1
        else:
            data[cls] += 1
    
    return data