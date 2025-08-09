# car_color_detection_app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(layout="wide", page_title="Car Colour Detection & Counting")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # small, fast; will auto-download if needed

model = load_model()

st.title("🚦 Car Colour Detection & People Count")
st.sidebar.header("Input")
input_mode = st.sidebar.selectbox("Input mode", ["Upload image", "Upload video", "Webcam (not on server)"])
uploaded = None
if input_mode == "Upload image":
    uploaded = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
elif input_mode == "Upload video":
    uploaded = st.sidebar.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
elif input_mode == "Webcam (local only)":
    st.sidebar.write("Webcam preview works only when running locally with a camera.")

st.sidebar.markdown("---")
st.sidebar.write("Options")
confidence = st.sidebar.slider("Detection confidence", 0.2, 0.9, 0.35)
iou = st.sidebar.slider("NMS IoU threshold", 0.3, 0.7, 0.45)

col1, col2 = st.columns([1,1])

def dominant_color_label(crop_bgr):
    # Convert to HSV and compute mean hue weighted by saturation/val
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Mask low-value pixels (dark)
    mask = v > 40
    if np.count_nonzero(mask) < 50:
        return "unknown"
    # Compute average hue over masked pixels
    avg_h = np.mean(h[mask])
    # Hue ranges (OpenCV 0-179)
    # blue roughly 90-130 (approx)
    if 90 <= avg_h <= 130:
        return "blue"
    # red spans around 0-10 and 160-179, but we only need to identify blue vs other
    return "other"

def process_image(image_np):
    # Run YOLO detection
    results = model(image_np, imgsz=640, conf=confidence, iou=iou, verbose=False)[0]
    boxes = []
    people_count = 0
    # map COCO class ids to names (ultralytics result has .names)
    names = results.names
    # Iterate detections
    if results.boxes is None:
        return image_np, 0, []
    for box in results.boxes:
        cls = int(box.cls.cpu().numpy())
        label = names[cls]
        conf = float(box.conf.cpu().numpy())
        xyxy = box.xyxy.cpu().numpy().astype(int)[0]  # [x1,y1,x2,y2]
        boxes.append((label, conf, xyxy))
    # Draw on copy
    vis = image_np.copy()
    car_infos = []
    for label, conf, (x1, y1, x2, y2) in boxes:
        if label == "person":
            people_count += 1
            # draw person with green bounding box and count later (not a hard requirement)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(vis, f"person {conf:.2f}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        elif label in ("car", "truck", "bus", "motorbike"):  # treat similar vehicle classes as cars optionally
            # crop safely
            h,w = image_np.shape[:2]
            xx1,yy1,xx2,yy2 = max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)
            crop = image_np[yy1:yy2, xx1:xx2]
            if crop.size == 0:
                color_label = "unknown"
            else:
                color_label = dominant_color_label(crop)
            # According to task: show red rectangle for blue cars, and blue rectangles for other colour cars.
            if color_label == "blue":
                rect_color = (0,0,255)   # BGR red
            else:
                rect_color = (255,0,0)   # BGR blue
            cv2.rectangle(vis, (x1,y1), (x2,y2), rect_color, 3)
            cv2.putText(vis, f"{label} | {color_label} | {conf:.2f}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, rect_color, 1)
            car_infos.append({"bbox":(x1,y1,x2,y2), "color":color_label, "conf":conf})
    return vis, people_count, car_infos

def np_from_file(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

if input_mode == "Upload image" and uploaded is not None:
    image_np = np_from_file(uploaded)
    vis, people_count, car_infos = process_image(image_np)
    st.sidebar.success(f"People detected: {people_count} | Cars detected: {len(car_infos)}")
    col1.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption="Input image", use_column_width=True)
    col2.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Detections", use_column_width=True)
    # summary table
    if car_infos:
        st.write("### Car details")
        for i,ci in enumerate(car_infos,1):
            st.write(f"{i}. Color: **{ci['color']}**, Confidence: {ci['conf']:.2f}, BBox: {ci['bbox']}")

elif input_mode == "Upload video" and uploaded is not None:
    tfile = uploaded
    # Save temp and process first frame as preview
    video_bytes = tfile.read()
    video_stream = cv2.VideoCapture(io.BytesIO(video_bytes))  # Not supported — fallback to saving file
    with open("temp_input_video.mp4", "wb") as f:
        f.write(video_bytes)
    cap = cv2.VideoCapture("temp_input_video.mp4")
    ret, frame = cap.read()
    if not ret:
        st.error("Can't read video. Try another file.")
    else:
        vis, people_count, car_infos = process_image(frame)
        st.sidebar.success(f"People detected (first frame): {people_count} | Cars detected: {len(car_infos)}")
        col1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="First frame", use_column_width=True)
        col2.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Detections (first frame)", use_column_width=True)
    cap.release()
else:
    st.write("Upload an image or video from the sidebar to get started.")
    st.write("Tip: run locally (`streamlit run car_color_detection_app.py`) to use webcam and GPU if available.")
