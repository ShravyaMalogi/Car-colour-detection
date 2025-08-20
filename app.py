import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import tempfile
import os
from PIL import Image

# Page config
st.set_page_config(page_title="Car & People Detection", layout="wide")

# Load CSS from style.css
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ================= SETTINGS =================
YOLO_WEIGHTS = "yolo/yolov8s.pt"             # YOLO model
COLOR_MODEL_PATH = "model/blue_car_classifier.keras"  # Car color classifier
COLOR_CLASSES = ["blue", "not_blue"]         # only 2 classes
BLUE_CONF_THRESHOLD = 0.75                   # Confidence threshold for blue detection
MIN_BOX_SIZE = 50                             # Ignore tiny detections
# If your sigmoid outputs "not_blue" as positive, flip this:
SIGMOID_POS_CLASS = "not_blue"
# ============================================

st.title("🚗 Car Color Detection (with People-count)")

uploaded_file = st.file_uploader("Upload an image or video", 
                                 type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])
process_button = st.button("Process File")

# ---------- helper for color classification ----------
def classify_car_color(car_crop_bgr, color_model, model_input_shape, threshold=0.9):
    car_img = cv2.cvtColor(car_crop_bgr, cv2.COLOR_BGR2RGB)
    car_img = cv2.resize(car_img, model_input_shape)
    car_img = car_img.astype("float32") / 255.0
    car_img = np.expand_dims(car_img, axis=0)

    pred = color_model.predict(car_img, verbose=0)

    if pred.shape[-1] == 1:
        # sigmoid: one output
        p = float(pred[0][0])
        if SIGMOID_POS_CLASS == "blue":
            p_blue = p
        else:
            p_blue = 1.0 - p
    else:
        # softmax: multi-class
        blue_index = COLOR_CLASSES.index("blue") if "blue" in COLOR_CLASSES else 0
        p_blue = float(pred[0][blue_index])

    label = "blue" if p_blue >= threshold else "not_blue"
    return label, p_blue
# -----------------------------------------------------

if uploaded_file and process_button:
    with st.spinner("Loading models..."):
        color_model = load_model(COLOR_MODEL_PATH)
        model_input_shape = tuple(color_model.input_shape[1:3])
        yolo_model = YOLO(YOLO_WEIGHTS)

    file_extension = uploaded_file.name.split(".")[-1].lower()
    is_video = file_extension in ["mp4", "avi", "mov", "mkv"]

    col1, col2 = st.columns(2)  # 👈 side-by-side layout

    if is_video:
        # ================= VIDEO PROCESSING =================
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        input_path = tfile.name

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            st.error("Error: Could not open uploaded video.")
        else:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            progress_bar = st.progress(0)
            frame_placeholder = col2.empty()  # processed output goes right
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_no = 0

            with col1:  # left = input preview
                st.video(input_path)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = yolo_model(frame, verbose=False)[0]
                frame_people, frame_cars = 0, 0

                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1

                    if cls_id == 0:  # Person
                        frame_people += 1

                    elif cls_id == 2 and w > MIN_BOX_SIZE and h > MIN_BOX_SIZE:  # Car
                        frame_cars += 1
                        car_crop = frame[y1:y2, x1:x2]
                        if car_crop.size > 0:
                            try:
                                color_label, p_blue = classify_car_color(
                                    car_crop, color_model, model_input_shape, BLUE_CONF_THRESHOLD
                                )
                            except Exception as e:
                                print(f"[WARNING] Prediction failed: {e}")
                                continue

                            box_color = (0, 0, 255) if color_label == "blue" else (255, 0, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(frame, f"{color_label} p={p_blue:.2f}", (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                cv2.putText(frame, f"People: {frame_people}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Cars: {frame_cars}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                out.write(frame)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                frame_no += 1
                progress_bar.progress(frame_no / frame_count)

            cap.release()
            out.release()

            st.success("Processing complete!")
            col2.video(output_path)
            st.download_button("Download Processed Video", data=open(output_path, "rb").read(),
                               file_name="processed_video.mp4", mime="video/mp4")

    else:
        # ================= IMAGE PROCESSING =================
        img = Image.open(uploaded_file)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        results = yolo_model(img_cv, verbose=False)[0]
        frame_people, frame_cars = 0, 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            if cls_id == 0:  # Person
                frame_people += 1

            elif cls_id == 2 and w > MIN_BOX_SIZE and h > MIN_BOX_SIZE:  # Car
                frame_cars += 1
                car_crop = img_cv[y1:y2, x1:x2]
                if car_crop.size > 0:
                    try:
                        color_label, p_blue = classify_car_color(
                            car_crop, color_model, model_input_shape, BLUE_CONF_THRESHOLD
                        )
                    except Exception as e:
                        print(f"[WARNING] Prediction failed: {e}")
                        continue

                    box_color = (0, 0, 255) if color_label == "blue" else (255, 0, 0)
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(img_cv, f"{color_label} p={p_blue:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        cv2.putText(img_cv, f"People: {frame_people}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img_cv, f"Cars: {frame_cars}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        st.success("Processing complete!")

        with col1:  # left column = input
            st.image(img, caption="Input", use_container_width=True)

        with col2:  # right column = processed
            st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB),
                     caption="Processed", use_container_width=True)

        st.download_button("Download Processed Image", 
                           data=cv2.imencode(".jpg", img_cv)[1].tobytes(),
                           file_name="processed_image.jpg", mime="image/jpeg")
