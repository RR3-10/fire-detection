import tempfile
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ---------- Page setup ----------
st.set_page_config(page_title="Fire & Smoke Detection", page_icon="🔥", layout="wide")
st.title("🔥 Fire & Smoke Detection")
st.caption("YOLO26n fine-tuned on the D-Fire dataset · test mAP50 = 0.775")

# ---------- Load the model ONCE ----------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------- Sidebar settings ----------
st.sidebar.header("⚙️ Settings")
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
alert_frames = st.sidebar.slider("Alert after N frames with fire", 1, 30, 5)
clear_frames = st.sidebar.slider("Clear after N frames without fire", 1, 60, 15)
mode = st.sidebar.radio("Input type", ["Image", "Video"])

# ---------- IMAGE mode ----------
if mode == "Image":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file:
        # uploaded file -> bytes -> OpenCV image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        result = model(img, conf=conf, verbose=False)[0]

        left, right = st.columns(2)
        left.image(img, channels="BGR", caption="Original")
        right.image(result.plot(), channels="BGR", caption="Detections")

        if len(result.boxes) > 0:
            found = [f"{model.names[int(b.cls)]} ({float(b.conf):.2f})" for b in result.boxes]
            st.error("🚨 Detected: " + ", ".join(found))
        else:
            st.success("✅ No fire or smoke detected")

# ---------- VIDEO mode ----------
else:
    file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if file:
        # OpenCV needs a real file path, so save the upload to a temporary file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(file.read())
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        status_box = st.empty()     # placeholder we update every frame
        frame_box = st.empty()      # placeholder for the video frame

        streak = miss = 0
        alert_on = False
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model(frame, conf=conf, verbose=False)[0]

            # same hysteresis logic as video_detect.py
            if len(result.boxes) > 0:
                streak += 1
                miss = 0
            else:
                streak = 0
                miss += 1
            if not alert_on and streak >= alert_frames:
                alert_on = True
            elif alert_on and miss >= clear_frames:
                alert_on = False

            if alert_on:
                status_box.error("🚨 FIRE ALERT")
            elif streak > 0:
                status_box.warning(f"Checking... {streak}/{alert_frames}")
            else:
                status_box.success("✅ Clear")
            frame_box.image(result.plot(), channels="BGR")

        cap.release()
        st.info("Video finished.")