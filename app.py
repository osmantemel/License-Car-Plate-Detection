import streamlit as st
import numpy as np
import cv2
import easyocr
import pandas as pd
from ultralytics import YOLO
import os
from streamlit_webrtc import webrtc_streamer
import av
from PIL import Image

# Model ve diğer ayarları
folder_path = "./licenses_plates_imgs_detected/"
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
COCO_MODEL_DIR = "./models/yolov8n.pt"

reader = easyocr.Reader(['en'], gpu=False)
vehicles = [2]

coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

threshold = 0.15

# Streamlit oturum durumu
if "state" not in st.session_state:
    st.session_state["state"] = "Uploader"

class VideoProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_to_an = img.copy()
        img_to_an = cv2.cvtColor(img_to_an, cv2.COLOR_RGB2BGR)
        license_detections = license_plate_detector(img_to_an)[0]

        if len(license_detections.boxes.cls.tolist()) != 0:
            for license_plate in license_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

                license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

                cv2.rectangle(img, (int(x1) - 40, int(y1) - 40), (int(x2) + 40, int(y1)), (255, 255, 255), cv2.FILLED)
                cv2.putText(img,
                            str(license_plate_text),
                            (int((int(x1) + int(x2)) / 2) - 70, int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 0),
                            3)

        # Anlık olarak video akışında kareyi dönüştür
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.readtext(license_plate_crop)

    if detections == []:
        return None, None

    plate = []

    for result in detections:
        text = result[1]
        text = text.upper()
        scores += result[2]
        plate.append(text)

    if len(plate) != 0:
        return " ".join(plate), scores / len(plate)
    else:
        return " ".join(plate), 0

def model_prediction(img):
    license_numbers = 0
    results = {}
    licenses_texts = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    object_detections = coco_model(img)[0]
    license_detections = license_plate_detector(img)[0]

    if len(object_detections.boxes.cls.tolist()) != 0:
        for detection in object_detections.boxes.data.tolist():
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection

            if int(class_id) in vehicles:
                cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)

    if len(license_detections.boxes.cls.tolist()) != 0:
        license_plate_crops_total = []
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

            licenses_texts.append(license_plate_text)

            if license_plate_text is not None and license_plate_text_score is not None:
                license_plate_crops_total.append(license_plate_crop)
                results[license_numbers] = {}

                results[license_numbers][license_numbers] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2], 'car_score': car_score},
                                                            'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                              'text': license_plate_text,
                                                                              'bbox_score': score,
                                                                              'text_score': license_plate_text_score}}
                license_numbers += 1

        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return [img_wth_box, licenses_texts, license_plate_crops_total]

    else:
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return [img_wth_box]

def change_state_uploader():
    st.session_state["state"] = "Uploader"

def change_state_camera():
    st.session_state["state"] = "Camera"

def change_state_live():
    st.session_state["state"] = "Live"

def change_state_video():
    st.session_state["state"] = "Video"

with st.container():
    _, col1, _ = st.columns([0.2, 1, 0.1])
    col1.title("💥 License Car Plate Detection 🚗")

    _, col4, _ = st.columns([0.1, 1, 0.2])

    _, col, _ = st.columns([0.3, 1, 0.2])
    col.image("./imgs/back1.png")

    _, col5, _ = st.columns([0.05, 1, 0.1])

with st.container():
    _, col1, _ = st.columns([0.1, 1, 0.2])

    _, colb1, colb2, colb3, colb4 = st.columns([0.2, 0.7, 0.6, 0.6, 1])

    if colb1.button("Upload an Image", on_click=change_state_uploader):
        pass
    elif colb2.button("Take a Photo", on_click=change_state_camera):
        pass
    elif colb3.button("Live Detection", on_click=change_state_live):
        pass
    elif colb4.button("Upload a Video", on_click=change_state_video):
        pass

    if st.session_state["state"] == "Uploader":
        img = st.file_uploader("Upload a Car Image: ", type=["png", "jpg", "jpeg"])
        if img is not None:
            image = np.array(Image.open(img))
            st.image(image, width=400)
            
            if st.button("Apply Detection"):
                results = model_prediction(image)
                if len(results) == 3:
                    prediction, texts, license_plate_crop = results[0], results[1], results[2]
                    
                    texts = [i for i in texts if i is not None]
                    
                    if len(texts) == 1 and len(license_plate_crop):
                        st.image(prediction)
                        st.image(license_plate_crop[0], width=350)
                        st.success(f"License Number: {texts[0]}")
                        
                        st.write("Detection results:")
                        st.write(results)
                        
                    elif len(texts) > 1 and len(license_plate_crop) > 1:
                        st.image(prediction)
                        for i in range(len(license_plate_crop)):
                            st.image(license_plate_crop[i], width=350)
                            st.success(f"License Number {i}: {texts[i]}")
                        
                        st.write("Detection results:")
                        st.write(results)
                else:
                    st.image(results[0])
                    
    elif st.session_state["state"] == "Camera":
        webrtc_streamer(key="camera", video_processor_factory=VideoProcessor)

    elif st.session_state["state"] == "Live":
        webrtc_streamer(key="live", video_processor_factory=VideoProcessor)

    elif st.session_state["state"] == "Video":
        video = st.file_uploader("Upload a Video: ", type=["mp4", "mov"])
        if video is not None:
            video_path = os.path.join("temp", "uploaded_video.mp4")
            with open(video_path, "wb") as f:
                f.write(video.read())

            # Videoyu okuma ve işleme
            cap = cv2.VideoCapture(video_path)
            frames = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Her kareyi işleyin
                results = model_prediction(frame)

                if len(results) > 0:
                    prediction = results[0]
                    frames.append(cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB))

            cap.release()
            os.remove(video_path)

            # Videoyu gösterim için streamlit video bileşeni kullanımı
            if frames:
                video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_file.name, fourcc, 20.0, (frames[0].shape[1], frames[0].shape[0]))

                for frame in frames:
                    out.write(frame)

                out.release()
                st.video(video_file.name)
