import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from collections import deque
from twilio.rest import Client

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
# account_sid = os.getenv("TWILIO_ACCOUNT_SID")
# auth_token = os.getenv("TWILIO_AUTH_TOKEN")
account_sid = "ACed4ed5e0907494ea975129f57d09de72";
auth_token = "4371c431d03ec576a9102ad453026d6e";

client = Client(account_sid, auth_token)

token = client.tokens.create()

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
    )
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
    )

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

def extract_keypoints(results):
    if results.pose_landmarks:
        keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        keypoints = np.zeros(33 * 4)
    padded_keypoints = np.zeros((30, 1662))
    num_keypoints = min(keypoints.shape[0], padded_keypoints.shape[1])
    padded_keypoints[:, :num_keypoints] = keypoints[:num_keypoints]
    return padded_keypoints

model = load_model('action.h5')
actions = np.array(["hello", "thanks", "please", "iloveyou", "takecare"])
colors = [(245, 39, 16), (108, 245, 16), (16, 184, 245), (245, 226, 16), (153, 16, 245)]

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model('action.h5')
        self.actions = np.array(["hello", "thanks", "please", "iloveyou", "takecare"])
        self.colors = [(245, 39, 16), (108, 245, 16), (16, 184, 245), (245, 226, 16), (153, 16, 245)]
        
        self.min_detection_confidence = 0.2
        self.min_tracking_confidence = 0.2
        
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.sentence_queue = deque(maxlen=6)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        image, results = mediapipe_detection(image, self.holistic)
        draw_landmarks(image, results)

        keypoints = extract_keypoints(results)
        res = self.model.predict(np.expand_dims(keypoints, axis=0))[0]

        image = prob_viz(res, self.actions, image, self.colors)

        if len(self.sentence_queue) > 0:  
            sentence_text = ' '.join(self.sentence_queue)
            cv2.rectangle(image, (0, 0), (image.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(image, sentence_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if np.max(res) > 0.6:
            action = self.actions[np.argmax(res)]
            if action not in self.sentence_queue:
                self.sentence_queue.append(action)

        return av.VideoFrame.from_ndarray(image, format='bgr24')

rtc_configuration = RTCConfiguration(
    {
        "iceServers": token.ice_servers
    }
)


def main():
    st.title("Real-Time Sign Detection")
    st.write("Using TensorFlow, Mediapipe, and OpenCV")

    webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        rtc_configuration=rtc_configuration
    )

if __name__ == '__main__':
    main()