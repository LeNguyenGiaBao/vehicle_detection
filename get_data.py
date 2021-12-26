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

if __name__ == "__main__":
    time_sleep = 15
    id_cameras = {
        '5d8cd542766c880017188948': 'Vo Van Ngan - Dang Van Bi',
        '5d8cd653766c88001718894c': 'Kha Van Can - Vo Van Ngan',
        '5d8cdb9f766c880017188968': 'Hoang Van Thu - Tran Huy Lieu',
        '5d8cdb0a766c880017188964': 'Truong Chinh - Xuan Hong',
        '5d8cdc57766c88001718896e': 'Ly Thuong Kiet - Lac Long Quan',
        '586e25e1f9fab7001111b0ae': 'Truong Chinh - Tan Ky Tan Quy',
        '5d8cd614766c88001718894a': 'Le Quang Dinh - No Trang Long',
        '5d8cdc9d766c880017188970': 'Lac Long Quan - Au Co',
        # '5fcddb85a461de001633356d': 'Bad - No Trang Long - Nguyen Huy Luong',
        # '5fcdd89fa461de0016333558': 'Bad - Nguyen Huy Canh - Ngo Tat To 2',
        # '58744e97b807da0011e33cb9': 'Bad - Duong Ba Trac - Hem 219'
        # '5ad0679598d8fc001102e274': 'Bad - Le Van Viet - Man Thien',
        # '5a606c078576340017d06624': 'Bad - Quoc Lo 13 - Hiep Binh',
        # '586e1f18f9fab7001111b0a5': 'Bad - Cong Hoa - Truong Chinh',
        # '5a606a958576340017d06621': 'Bad - Dinh Bo Linh - Bach Dang',
        # '5f02d853942cda00169ee0a0': 'Bad - Le Van Viet - Hoang Huu Nam',
    }
    while True:
        for id in id_cameras:
            img = get_image(id)
            current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            cv2.imwrite('../data/' + current_time + '.jpg', img)
            print(current_time)
            time.sleep(time_sleep/len(id_cameras))
