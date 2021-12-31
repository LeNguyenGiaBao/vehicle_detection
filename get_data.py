import cv2 
import time
from cameras import cameras
from utils.utils import get_image, get_cur_time

if __name__ == "__main__":
    time_sleep = 15

    while True:
        for camera in cameras:
            img = get_image(camera['id_camera'])
            current_time = get_cur_time()
            cv2.imwrite('../data/' + current_time + '.jpg', img)
            print(current_time)
            time.sleep(time_sleep/len(cameras))
