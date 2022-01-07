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

    b11, b12, b13, b14, b15, b16, b17, b18 = st.columns((1,2,1.5,0.7,0.7,0.7,0.7,0.7))
    app_mode = b11.selectbox('Mode',
        ('About Page','Run on Image','Run on Live Camera', 'Run on Video')
        )

    if app_mode =='About Page':
        img_result = cv2.imread('./data/result.jpg')
        img_result = change_color_and_size(img_result)
        img_slack = cv2.imread('./image/slack_2.png')
        img_slack = change_color_and_size(img_slack)
        img_live_camera = cv2.imread('./image/live_camera.png')
        img_live_camera = change_color_and_size(img_live_camera)
        
        _, b21 = st.columns((1,10))
        b21.markdown("<p style='text-align: left; color: black; font-size: 25px;'>The website is used to identify vehicles, assess and visualize traffic conditions in Ho Chi Minh City, Vietnam</p>", unsafe_allow_html=True)
        b21.markdown("<p style='text-align: left; color: black; font-size: 20px;'>Data from <a href='http://giaothong.hochiminhcity.gov.vn' target='_blank'>Cổng Thông Tin Vận Tải Thành Phố Hồ Chí Minh </a></p>", unsafe_allow_html=True)
        st.markdown('---')

        _, b21 = st.columns((1,10))
        b21.markdown("## Application")

        _, b21, _ = st.columns((1,3,1))
        b21.markdown('---')
        _, b21, b22, b23, _ = st.columns((1,1,3,1,1))
        b21.markdown("""<p style='text-align: left; color: red; font-size: 25px;'> Detect Vehicles
        <ul>
            <li style='text-align: left; color: black; font-size: 20px;'>From Image</li>
            <li style='text-align: left; color: black; font-size: 20px;'>From Video</li>
            <li style='text-align: left; color: black; font-size: 20px;'>From Live Camera</li>
        </ul>
        </p>""", unsafe_allow_html=True)
        b22.image(img_result)

        _, b21, _ = st.columns((1,3,1))
        b21.markdown('---')
        _, b21, b22, _, b23 = st.columns((1,1,3,0.2, 1.8))
        b22.image(img_live_camera)
        b23.markdown("""<p style='text-align: left; color: red; font-size: 25px;'> Visualize Information With Charts </p>""", unsafe_allow_html=True)
        b23.markdown("""<p style='text-align: left; color: black; font-size: 20px;'> Using Realtime Information To Draw Charts </p>""", unsafe_allow_html=True)
        _, b21, _ = st.columns((1,3,1))
        b21.markdown('---')
        _, b21, b22, b23, _ = st.columns((0.5, 1.5,3,1,1))
        b22.image(img_slack)
        b21.markdown("""<p style='text-align: left; color: red; font-size: 25px;'> Notifications When Traffic Jams </p>""", unsafe_allow_html=True)
        b21.markdown("""<p style='text-align: left; color: black; font-size: 20px;'> Join Slack Channel <a href='https://app.slack.com/client/T02RPGAG9D5/C02RXEK7806'>Here</a> With Us</p>""", unsafe_allow_html=True)
        
        st.markdown('---')
        _, b21 = st.columns((1,10))
        b21.markdown("## About Us")
        st.markdown('')
        b31, b32, b33, b34, b35 = st.columns((1,3, 0.1,3,1))
        b32.markdown('<p align="center"><img src="https://avatars.githubusercontent.com/u/68860804?v=4" width="300px" /></p>', unsafe_allow_html=True)
        b32.markdown("<h2 style='text-align: center; color: black;'>Lê Nguyễn Gia Bảo</h1>", unsafe_allow_html=True)
        b32.markdown("<h3 style='text-align: center; color: black;'>18110251</h2>", unsafe_allow_html=True)
        b32.markdown('''
        <p align="center">
            <a href="https://www.linkedin.com/in/lenguyengiabao/" target="_blank">
                <img src="https://img.icons8.com/fluent/48/000000/linkedin.png"/>
            </a>
            <a href="https://www.facebook.com/baorua.98/" alt="Facebook" target="_blank">
                <img src="https://img.icons8.com/fluent/48/000000/facebook-new.png" />
            </a> 
            <a href="https://github.com/LeNguyenGiaBao" alt="Github" target="_blank">
                <img src="https://img.icons8.com/fluent/48/000000/github.png"/>
            </a> 
            <a href="https://www.youtube.com/channel/UCOZbUfO_au3oxHEh4x52wvw/videos" alt="Youtube channel" target="_blank" >
                <img src="https://img.icons8.com/fluent/48/000000/youtube-play.png"/>
            </a>
            <a href="https://www.kaggle.com/nguyngiabol" alt="Kaggle" target="_blank" >
                <img src="https://img.icons8.com/windows/48/000000/kaggle.png"/>
            </a>
            <a href="mailto:lenguyengiabao46@gmail.com" alt="Email" target="_blank">
                <img src="https://img.icons8.com/fluent/48/000000/mailing.png"/>
            </a>
        </p>
        ''', unsafe_allow_html=True)

        b34.markdown('<p align="center"><img src="https://scontent.fsgn13-2.fna.fbcdn.net/v/t1.6435-9/50291308_2508028652757436_5546464184155242496_n.jpg?_nc_cat=108&ccb=1-5&_nc_sid=09cbfe&_nc_ohc=ZZz3SZMaXYgAX-881LS&_nc_ht=scontent.fsgn13-2.fna&oh=00_AT80newat36TZg94zbZLK_df5OFmoCo-VDKx2vKGe7s9zQ&oe=61FB6199" width="300px" /></p>', unsafe_allow_html=True)
        b34.markdown("<h2 style='text-align: center; color: black;'>Trần Trung Kiên</h1>", unsafe_allow_html=True)
        b34.markdown("<h3 style='text-align: center; color: black;'>18110309</h2>", unsafe_allow_html=True)
        b34.markdown('''
        <p align="center">
            <a href="https://www.linkedin.com/in/lenguyengiabao/" target="_blank">
                <img src="https://img.icons8.com/fluent/48/000000/linkedin.png"/>
            </a>
            <a href="https://www.facebook.com/trantrungkien2035" alt="Facebook" target="_blank">
                <img src="https://img.icons8.com/fluent/48/000000/facebook-new.png" />
            </a> 
            <a href="https://github.com/ttkien2035" alt="Github" target="_blank">
                <img src="https://img.icons8.com/fluent/48/000000/github.png"/>
            </a> 
            <a href="https://www.youtube.com/channel/UCOZbUfO_au3oxHEh4x52wvw/videos" alt="Youtube channel" target="_blank" >
                <img src="https://img.icons8.com/fluent/48/000000/youtube-play.png"/>
            </a>
            <a href="https://www.kaggle.com/nguyngiabol" alt="Kaggle" target="_blank" >
                <img src="https://img.icons8.com/windows/48/000000/kaggle.png"/>
            </a>
            <a href="mailto:trantrungkien2035@gmail.com" alt="Email" target="_blank">
                <img src="https://img.icons8.com/fluent/48/000000/mailing.png"/>
            </a>
        </p>
        ''', unsafe_allow_html=True)

        _, b21 = st.columns((1,10))
        b21.markdown("## From ")
        _, b21, b22 = st.columns((2,2, 7))
        b21.markdown("<p style='text-align: center; color: black; font-size: 20px;'><img src='http://hcmute.edu.vn/Resources/Images/Logo/Logo%20HCMUTE-Corel-white%20background.jpg' width='150px' /> </p>", unsafe_allow_html=True)
        b22.markdown('')
        b22.markdown('')
        b22.markdown('')
        b22.markdown("<p style='text-align: left; color: black; font-size: 30px;'>Ho Chi Minh City University of Technology and Education</p>", unsafe_allow_html=True)
        b22.markdown("<p style='text-align: left; color: black; font-size: 20px;'><a href='https://hcmute.edu.vn'>https://hcmute.edu.vn</a></p>", unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')
        _, b21, b22 = st.columns((2,2, 7))
        b21.markdown("<p style='text-align: center; color: black; font-size: 20px;'><img src='https://fit.hcmute.edu.vn/Resources/Images/SubDomain/fit/logo-cntt2021.png' width='150px' /> </p>", unsafe_allow_html=True)
        b22.markdown('')
        b22.markdown("<p style='text-align: left; color: black; font-size: 30px;'>Faculty Of Information Technology</p>", unsafe_allow_html=True)
        b22.markdown("<p style='text-align: left; color: black; font-size: 20px;'><a href='fit.hcmute.edu.vn'>https://fit.hcmute.edu.vn</a></p>", unsafe_allow_html=True)

        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown("<p style='text-align: center; color: black; font-size: 20px;'>Design By <a href='https://streamlit.io' target='_blank'>Streamlit</a></p>", unsafe_allow_html=True)

        
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
        space_graph = b22.line_chart(df, height=480, use_container_width=True)
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
                
                space_graph.line_chart(df,  height=480, use_container_width=True)
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


@st.cache
def load_model(model_path, class_names):
    net = SSD(len(class_names), is_test=True)
    net.load(model_path)
    model = Predictor(net, nms_method='soft', candidate_size=200)
    
    return model

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
    model = load_model(model_path, class_names)
    UI(model, class_names)
