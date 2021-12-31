import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import io
import pandas as pd
from model import SSD, Predictor
from utils.utils import get_image, count_number_per_class, draw_boxes
from cameras import cameras
from tornado import httpclient


def sleep_n(n):
    global stop_sleep
    count = 0
    while count < n:
        if stop_sleep:
            stop_sleep = False
            break
        time.sleep(1)
        count += 1


def predict(img, model, class_names, thresh, get_data=False):
    boxes, labels, probs = model.predict(img, 50, thresh)
    img = draw_boxes(img, boxes, labels, probs, class_names)

    if get_data:
        data = count_number_per_class(class_names, labels)

    return img, data
  

def change_color_and_size(img):
    h, w, c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1000, int(1000 * h / w)))
    
    return img 


def UI(model, class_names):
    st.set_page_config(page_title='Traffic Monitoring Application ', layout="wide")
    st.markdown("<h1 style='text-align: center; color: black;'>Traffic Monitoring Application</h1>", unsafe_allow_html=True)
    # st.title('Traffic Monitoring Application')

    b11, b12, b13, b14, b15, b16, b17, b18 = st.columns((1,2,1.5,0.7,0.7,0.7,0.7,0.7))
    app_mode = b11.selectbox('Mode',
        ('About Page','Run on Image','Run on Live Camera', 'Run on Video')
        )

    if app_mode =='About Page':
        st.markdown('**StreamLit** is to create the Web Graphical User Interface (GUI) ')
    
        
    elif app_mode =='Run on Image':
        img_file_buffer = b12.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])
        detection_confidence = b13.slider('Detection Confidence', min_value =1,max_value = 99,value = 50)        
        threshold = detection_confidence/100
        b15.markdown('**Motorcycle**')
        moto_value = b15.markdown('0')

        b16.markdown('**Car**')
        car_value = b16.markdown('0')

        b17.markdown('**Bus**')
        bus_value = b17.markdown('0')

        b18.markdown('**Truck**')
        truck_value = b18.markdown('0')

        space_img = st.empty()
        if img_file_buffer is not None:
            img = np.array(Image.open(img_file_buffer))
            img_pred, data = predict(img, model, class_names, threshold, get_data=True)
            height, width, _ = img_pred.shape
            moto_value.markdown(data['motorcycle'])
            car_value.markdown(data['car'])
            bus_value.markdown(data['bus'])
            truck_value.markdown(data['truck'])
            img_pred = cv2.resize(img_pred, (1000, int(1000 * height / width)))
            space_img.image(img_pred)
        
            

    elif app_mode=='Run on Live Camera':
        b21, b22 = st.columns((3,2))
        df = pd.DataFrame([[0,0,0,0]]*20, columns=['motorcycle', 'car', 'bus', 'truck'])
        id_camera_selectbox = b12.selectbox('Id Camera', [i['name'] for i in id_cameras])
        id_camera = id_cameras[[i['name'] for i in id_cameras].index(id_camera_selectbox)]['id_camera']
        detection_confidence = b13.slider('Detection Confidence', min_value =1,max_value = 99,value = 50)

        threshold = detection_confidence/100
        b15.markdown('**Motorcycle**')
        moto_value = b15.markdown('0')

        b16.markdown('**Car**')
        car_value = b16.markdown('0')

        b17.markdown('**Bus**')
        bus_value = b17.markdown('0')

        b18.markdown('**Truck**')
        truck_value = b18.markdown('0')
        space_img = b21.empty()
        space_graph = b22.line_chart(df, height=600, use_container_width=True)
        if id_camera != '':
            while True:
                img = get_image(id_camera)
                img_pred, data = predict(img, model, class_names, threshold, get_data=True)
                moto_value.markdown(data['motorcycle'])
                car_value.markdown(data['car'])
                bus_value.markdown(data['bus'])
                truck_value.markdown(data['truck'])
                height, width, _ = img_pred.shape
                img_pred = change_color_and_size(img_pred)
                space_img.image(img_pred)
                df_new = pd.DataFrame([data.values()], columns=['motorcycle', 'car', 'bus', 'truck'])
                df = df.iloc[1:, :]
                df = df.append(df_new, ignore_index=True)
                
                space_graph.line_chart(df,  height=600, use_container_width=True)
                sleep_n(15)

    
    elif app_mode == 'Run on Video':
        video_file_buffer = b12.file_uploader("Upload a video", type=[ "mp4", "mov",'avi' ])
        detection_confidence = b13.slider('Detection Confidence', min_value =1,max_value = 99,value = 50)        
        threshold = detection_confidence/100
        b15.markdown('**Motorcycle**')
        moto_value = b15.markdown('0')

        b16.markdown('**Car**')
        car_value = b16.markdown('0')

        b17.markdown('**Bus**')
        bus_value = b17.markdown('0')

        b18.markdown('**Truck**')
        truck_value = b18.markdown('0')
        st.set_option('deprecation.showfileUploaderEncoding', False)

        record = b14.checkbox("Record Video")
        
        tfflie = tempfile.NamedTemporaryFile(delete=False)
        space_img = st.empty()
        if video_file_buffer:
            tfflie.write(video_file_buffer.read())
            vid = cv2.VideoCapture(tfflie.name)

            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_input = int(vid.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out_video = cv2.VideoWriter('./data/output.avi',fourcc, 20.0, (width, height))
            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                    break

                img_pred, data = predict(frame, model, class_names, threshold, get_data=True)
                
                
                if record:
                    out_video.write(img_pred)

                img_pred = change_color_and_size(img_pred)
                space_img.image(img_pred)

            vid.release()
            out_video.release()
            
            if record:
                with open('./data/output.avi', 'rb') as f:
                    download_button = b14.download_button('Download', f, 'output.avi', mime="video/mp4")


if __name__ == "__main__":
    http_client = httpclient.HTTPClient()
    stop_sleep = False
    id_cameras = [
        {
            'name': 'Choose camera',
            'id_camera': ''
        }
    ] + cameras
    class_names = ['BACKGROUND', 'motorcycle', 'car', 'bus', 'truck']
    model_path = './models/vgg16-ssd-Epoch-170-Loss-1.8997838258743287.pth'
    net = SSD(len(class_names), is_test=True)
    net.load(model_path)
    model = Predictor(net, nms_method='soft', candidate_size=200)
    UI(model, class_names)
