import cv2 
import time
from cameras import cameras
from utils.utils import get_image, get_cur_time

if __name__ == "__main__":
    time_sleep = 15
    # id_cameras = {
    #     '5d8cd542766c880017188948': 'Vo Van Ngan - Dang Van Bi',
    #     '5d8cd653766c88001718894c': 'Kha Van Can - Vo Van Ngan',
    #     '5d8cdb9f766c880017188968': 'Hoang Van Thu - Tran Huy Lieu',
    #     '5d8cdb0a766c880017188964': 'Truong Chinh - Xuan Hong',
    #     '5d8cdc57766c88001718896e': 'Ly Thuong Kiet - Lac Long Quan',
    #     '586e25e1f9fab7001111b0ae': 'Truong Chinh - Tan Ky Tan Quy',
    #     '5d8cd614766c88001718894a': 'Le Quang Dinh - No Trang Long',
    #     '5d8cdc9d766c880017188970': 'Lac Long Quan - Au Co',
    #     # '5fcddb85a461de001633356d': 'Bad - No Trang Long - Nguyen Huy Luong',
    #     # '5fcdd89fa461de0016333558': 'Bad - Nguyen Huy Canh - Ngo Tat To 2',
    #     # '58744e97b807da0011e33cb9': 'Bad - Duong Ba Trac - Hem 219'
    #     # '5ad0679598d8fc001102e274': 'Bad - Le Van Viet - Man Thien',
    #     # '5a606c078576340017d06624': 'Bad - Quoc Lo 13 - Hiep Binh',
    #     # '586e1f18f9fab7001111b0a5': 'Bad - Cong Hoa - Truong Chinh',
    #     # '5a606a958576340017d06621': 'Bad - Dinh Bo Linh - Bach Dang',
    #     # '5f02d853942cda00169ee0a0': 'Bad - Le Van Viet - Hoang Huu Nam',
    # }
    while True:
        for camera in cameras:
            img = get_image(camera['id_camera'])
            current_time = get_cur_time()
            cv2.imwrite('../data/' + current_time + '.jpg', img)
            print(current_time)
            time.sleep(time_sleep/len(cameras))
