import streamlit as st
import cv2
import os
from detector import detect_phones, save_history
from PIL import Image
from datetime import timedelta
import io


FRAME_SKIP = 24

def process_video(source, is_file=True, filename="video_frame"):
    cap = cv2.VideoCapture(source if is_file else 0)
    frame_count = 0
    saved_frames = 0

    stframe = st.empty()
    st.info("🔍 Обработка видео...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        if frame_count % FRAME_SKIP == 0:
            buf = io.BytesIO()
            image_pil.save(buf, format="JPEG")
            img_bytes = buf.getvalue()

            img_annotated, num_detections = detect_phones(img_bytes)

            if num_detections > 0:
                saved_frames += 1
                seconds = int(cap.get(cv2.CAP_PROP_POS_MSEC)) // 1000
                timestamp = str(timedelta(seconds=seconds)).replace(":", "-")
                file_name = f"{filename}_{timestamp}.jpg"
                img_annotated.save(os.path.join("detections", file_name))
                save_history(file_name, True, num_detections)

                stframe.image(img_annotated, caption=f"📱 Найдено: {num_detections}", use_container_width=True)
            else:
                stframe.image(image_pil, caption="🚫 Телефонов не найдено", use_container_width=True)

    cap.release()
    st.success(f"🎉 Стрим завершён. Сохранено кадров с телефонами: {saved_frames}")


st.title("📱 Детектор использования телефонов на экзамене")

mode = st.radio("Выберите режим", ["Изображение", "Видео файл", "Веб-камера"])

if mode == "Изображение":
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Загруженное изображение", use_container_width=True)

        if st.button("🚀 Запустить детекцию"):
            img_bytes = uploaded_file.read()
            img, num_detections = detect_phones(img_bytes)

            if num_detections > 0:
                st.success(f"✅ Обнаружено телефонов: {num_detections}")
                save_history(uploaded_file.name, True, num_detections)
            else:
                st.info("🔍 Телефоны не обнаружены")

            st.image(img, caption="Обработанное изображение", use_container_width=True)

elif mode == "Видео файл":
    uploaded_video = st.file_uploader("Загрузите видеофайл", type=["mp4", "avi", "mov"])
    if uploaded_video:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        if st.button("🚀 Запустить обработку видео"):
            process_video("temp_video.mp4", is_file=True, filename=uploaded_video.name)

elif mode == "Веб-камера":
    if st.button("🎥 Начать стрим с камеры"):
        process_video(0, is_file=False, filename="webcam_frame")

if st.button("📥 Скачать отчёт (CSV)"):
    from detector import export_csv
    if export_csv():
        with open("report.csv", "rb") as f:
            st.download_button("Скачать CSV", data=f, file_name="report.csv", mime="text/csv")
    else:
        st.warning("⚠️ Пока нет данных для отчёта")
