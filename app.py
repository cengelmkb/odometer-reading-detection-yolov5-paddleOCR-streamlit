import glob
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time
import numpy as np
import re
from paddleocr import PaddleOCR
st.set_page_config(layout="wide")

ocr_model = PaddleOCR(lang='en') #Paddle OCR
cfg_model_path = 'last.pt'
model = None
confidence = .25

### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx (frame, model):
    frame = [frame]
    results = model(frame)

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame,classes):

    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels

    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]


    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55: ### threshold value for detection. We are discarding everything below this value
            
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]
            

            coords = [x1,y1,x2,y2]

            plate_num = recognize_plate_easyocr(img = frame, coords= coords)


            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
            cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, .7,(255,255,255), 2)

            
    return frame

#### ---------------------------- function to recognize license plate --------------------------------------


# function to recognize odometer reading using paddle OCR
def recognize_plate_easyocr(img, coords):
    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    #nplate = img[int(ymin)-7:int(ymax)+5, int(xmin)-7:int(xmax)+5]
    nplate = img[int(ymin)-5:int(ymax)+5, int(xmin)-10:int(xmax)+10] ### cropping the number plate from the whole image

    ocr_result = ocr_model.ocr(nplate)

    text = ocr_result

    if len(text) == 0:
        return text
    else: 
        if len(text[0]) == 0:
            return text
        else:
            if len(text[0][0]) == 0:
                return text
            else:
                if len(text[0][0][0]) == 0:
                    return text
                else:
                    if len(text[0][0][0][0]) == 0:
                        return text
                    else:
                        return text[0][0][1][0]






def image_input(data_src):
    img_file = None
    if data_src == 'Sample data':
        # get all sample images
        
        img_path = glob.glob('data/sample_images/*')
        #img_out_name = 'for_web.jpg'
        list_image = np.arange (1,len(img_path)+1,1)
        
        option = st.selectbox(
            'Select an sample image',
             list_image, index=0
        )
        #img_slider = st.slider("Select a test image.", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[option - 1]
        frame = cv2.imread(img_file) ### reading the image
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = detectx(frame, model = model) ### DETECTION HAPPENING HERE    

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        classes = model.names
        frame = plot_boxes(results, frame,classes = classes)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #result_img = cv2.imwrite(img_out_name,frame)
    else:
        img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)
            frame = cv2.imread(img_file) ### reading the image
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
            results = detectx(frame, model = model) ### DETECTION HAPPENING HERE    

            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            classes = model.names
            frame = plot_boxes(results, frame,classes = classes)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            img = frame
            #img = infer_image(img_file)
            st.image(img, caption="Model prediction")


def video_input(data_src):
    vid_file = None
    webcam_feed = None
    if data_src == 'Sample data':
        vid_file = "data/sample_videos/sample.mp4"
    elif data_src == 'Use Webcam':
        webcam_feed = cv2.VideoCapture(0)
        
    else:
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
            results = detectx(frame, model = model) ### DETECTION HAPPENING HERE    

            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            classes = model.names
            frame = plot_boxes(results, frame,classes = classes)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            #output_img = infer_image(frame)
            output.image(frame)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")

        cap.release()
    if webcam_feed:
        cap = cv2.VideoCapture(0)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
            results = detectx(frame, model = model) ### DETECTION HAPPENING HERE    

            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            classes = model.names
            frame = plot_boxes(results, frame,classes = classes)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            output.image(frame)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")

        cap.release()

def infer_image(img, size=None):
    model.conf = confidence
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image


#@st.experimental_singleton
@st.cache_resource #caches the model for faster reloads. Also avoids loading same model repeatedly
def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    print("model to ", device)
    return model_


#st.experimental_singleton
@st.cache_resource
def download_model(url):
    model_file = wget.download(url, out="models")
    return model_file


def get_user_model():
    model_src = st.sidebar.radio("Model source", ["file upload", "url"])
    model_file = None
    if model_src == "file upload":
        model_bytes = st.sidebar.file_uploader("Upload a model file", type=['pt'])
        if model_bytes:
            model_file = "models/uploaded_" + model_bytes.name
            with open(model_file, 'wb') as out:
                out.write(model_bytes.read())
    else:
        url = st.sidebar.text_input("model url")
        if url:
            model_file_ = download_model(url)
            if model_file_.split(".")[-1] == "pt":
                model_file = model_file_

    return model_file

def main():
    # global variables
    global model, confidence, cfg_model_path

    st.title("Odometer Detection using YOLOv5 and Reading extraction using PaddleOCR")

    st.sidebar.title("Select Parameters")

    # upload model
    model_src = st.sidebar.radio("Select yolov5 weight file", ["Use our Custom Trained Model", "Upload your own model"])
    # URL, upload file (max 200 mb)
    if model_src == "Use your own model":
        user_model_path = get_user_model()
        if user_model_path:
            cfg_model_path = user_model_path

        st.sidebar.text(cfg_model_path.split("/")[-1])
        st.sidebar.markdown("---")

    # check if model file is available
    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available!!!, please added to the model folder.", icon="⚠️")
    else:
        # device options
        if torch.cuda.is_available():
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=1)
        else:
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

        # load model
        model = load_model(cfg_model_path, device_option)

        # confidence slider
        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)

        # custom classes
        if st.sidebar.checkbox("Custom Classes"):
            model_names = list(model.names.values())
            assigned_class = st.sidebar.multiselect("Select Classes", model_names, default=[model_names[0]])
            classes = [model_names.index(name) for name in assigned_class]
            model.classes = classes
        else:
            model.classes = list(model.names.keys())

        st.sidebar.markdown("---")

        # input options
        input_option = st.sidebar.radio("Select input file type: ", ['Image', 'Video'])

        # input src option
        if input_option == 'Image':
            data_src = st.sidebar.radio("Select input file source ", ['Sample data', 'Upload your own data'])
        elif input_option == 'Video':
            data_src = st.sidebar.radio("Select input file source ", ['Sample data', 'Upload your own data','Use Webcam'])
        if input_option == 'Image':
            image_input(data_src)
        else:
            video_input(data_src)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
