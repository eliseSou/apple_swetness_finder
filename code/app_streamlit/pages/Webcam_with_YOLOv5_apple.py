import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import av
import pandas
from yolov5.utils.plots import Annotator, colors
from pathlib import Path
import os
import twilio
from twilio.rest import Client
from dotenv import load_dotenv

def load_yolov5_model(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    return model

class YOLOv5VideoTransformer(VideoProcessorBase):
    def __init__(self):
        model_path = Path("final.pt")
        self.model = load_yolov5_model(model_path)
        # self.model = torch.hub.load("ultralytics/yolov5", 'custom', pretrained=False)
        # self.model.load_state_dict(torch.load('yolov5s_apple.pt'))
        self.model.eval()
        self.names = self.model.names

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # input_img = self.preprocess(pil_img).unsqueeze(0)
        with torch.no_grad():
            results = self.model(img_rgb)

        labels = results.xyxy[0][:, -1].numpy()
        boxes = results.xyxy[0][:, :-1].numpy()

        annotator = Annotator(img_rgb)
        for i, (label, box) in enumerate(zip(labels, boxes)):
            class_name = self.names[int(label)]
            color = colors(int(label))
            annotator.box_label(box, f"{class_name}: {box[4]:.2f}", color=color)
        
        result_img = cv2.cvtColor(annotator.im, cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(result_img, format="bgr24")


st.header("Object Detection with YOLOv5")
st.markdown("Click the 'Start' button below to access your webcam and see the object detection in real-time.")

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
# load_dotenv()
# account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
# auth_token = os.environ.get('TWILIO_AUTH_TOKEN')

# client = Client(account_sid, auth_token)

# token = client.tokens.create()

webrtc_ctx = webrtc_streamer(key="YOLOv5", 
                            mode=WebRtcMode.SENDRECV,
                            video_processor_factory=YOLOv5VideoTransformer,
                            media_stream_constraints={"video": True, "audio": False},
                            async_processing=True,
                            rtc_configuration={
                            #     "iceServers": token.ice_servers
                                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                            }
                            )



# source ~/anaconda3/etc/profile.d/conda.sh
