import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

st.set_page_config(page_title="Helmet & Head Detection Demo", layout="wide")
st.title("🪖 Helmet & Head Detection Demo")

# 1️⃣ ONNX modelni yuklash
model_path = "runs/detect/helmet_train/weights/best.onnx"  # sizning ONNX manzilingiz
model = YOLO(model_path)

# 2️⃣ Foydalanuvchi video yuklashi
uploaded_file = st.file_uploader("Video yuklang (mp4 yoki avi)", type=["mp4", "avi"])

if uploaded_file is not None:
    # Faylni vaqtinchalik saqlash
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Video oqimini o‘qish
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Natija videoni saqlash uchun
    output_path = "output_annotated.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stframe = st.empty()
    st.text("Video annotatsiya qilinmoqda...")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)  # ONNX model bilan inference
        annotated_frame = results[0].plot()  # Annotatsiya

        out.write(annotated_frame)
        # Streamlit-da natijani ko‘rsatish (RGB format)
        stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    out.release()
    st.success("✅ Video tayyor! 'output_annotated.avi' faylini quyida yuklab oling.")

    # Yuklash tugmasi
    with open(output_path, "rb") as f:
        st.download_button("📥 Annotated videoni yuklab olish", f, file_name="annotated_video.avi")
