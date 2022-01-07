import numpy as np 
import cv2
import io
import time
import requests
from model import SSD, Predictor
from cameras import cameras
from get_data import get_image
from utils.utils import draw_boxes
from bot_token import SLACK_TOKEN

def send_message(img, name_camera):
    SLACK_URL = 'https://slack.com/api/files.upload'
    CHANNEL_ID = 'C02RXEK7806'
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        _, encoded_image = cv2.imencode('.jpeg', img, encode_param)

        files = {
            'file': io.BytesIO(encoded_image)
        }
        param = {
            'token': SLACK_TOKEN,
            'channels': CHANNEL_ID,
            'filename': "detected.jpeg",
            'initial_comment': "{} traffic jam!".format(name_camera),
            'title': "image"
        }

        response = requests.post(SLACK_URL, params=param, files=files)
        print(name_camera, response)
        
    except Exception as e:
        print('Could not send image to Slack: {}'.format(e))


def is_congestion(camera, sum_vehicle):
    if camera['previous_sum'] != -1 and sum_vehicle - camera['previous_sum'] > 2 and sum_vehicle > 5:
        return True

def monitor_traffic(cameras):
    while True:
        for camera in cameras:
            id_camera = camera['id_camera']
            img = get_image(id_camera)
            boxes, labels, probs = predictor.predict(img, 50, 0.3)
            if is_congestion(camera, len(boxes)):
                img_drawed = draw_boxes(img, boxes, labels, probs, class_names)
                send_message(img_drawed, camera['name'])
            
            camera['previous_sum'] = len(boxes)
            time.sleep(15/len(cameras))
            
if __name__ == "__main__":
    class_names = ['BACKGROUND', 'motorcycle', 'car', 'bus', 'truck']
    model_path = './models/vgg16-ssd-Epoch-170-Loss-1.8997838258743287.pth'
    net = SSD(len(class_names), is_test=True)
    net.load(model_path)
    predictor = Predictor(net, nms_method='soft', candidate_size=200)
    for i in cameras:
        i.update(previous_sum =-1)

    monitor_traffic(cameras)

