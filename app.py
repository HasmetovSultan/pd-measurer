import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="PD Measurer", page_icon="💳", layout="centered")

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage, ImageFormat

@st.cache_resource
def init_face_landmarker():
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5
    )
    return vision.FaceLandmarker.create_from_options(options)

landmarker = init_face_landmarker()

st.title("💳 Измерение PD по карте")
st.info("📐 Держи карту горизонтально у лица")

uploaded_file = st.file_uploader("Выберите фото", type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape
    
    # 🔹 Детекция лица
    mp_image = MPImage(image_format=ImageFormat.SRGB, data=img_np)
    detection_result = landmarker.detect(mp_image)
    
    if not detection_result.face_landmarks:
        st.error("❌ Лицо не найдено!")
        st.stop()
    
    st.success("✅ Лицо найдено!")
    landmarks = detection_result.face_landmarks[0]
    
    left_pupil = landmarks[468]
    right_pupil = landmarks[473]
    
    left_x = int(left_pupil.x * w)
    left_y = int(left_pupil.y * h)
    right_x = int(right_pupil.x * w)
    right_y = int(right_pupil.y * h)
    
    pd_px = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
    
    # 🔹 Инициализация
    if 'manual_x' not in st.session_state:
        st.session_state.manual_x = int(w * 0.7)
    if 'manual_y' not in st.session_state:
        st.session_state.manual_y = h // 2
    if 'manual_width' not in st.session_state:
        st.session_state.manual_width = int(w * 0.2)
    if 'zoom' not in st.session_state:
        st.session_state.zoom = 600
    
    manual_x = st.session_state.manual_x
    manual_y = st.session_state.manual_y
    manual_width = st.session_state.manual_width
    zoom = st.session_state.zoom
    
    # 🔹 МАКЕТ: Слева настройки, Справа фото
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("🎯 Настройка рамки")
        
        # X
        st.write("**↔️ Позиция X**")
        manual_x = st.slider("X", 0, w, manual_x, 3, key="x_slider")
        st.session_state.manual_x = manual_x
        st.write(f"`{manual_x} px`")
        
        # Y
        st.write("**↕️ Позиция Y**")
        manual_y = st.slider("Y", 0, h, manual_y, 3, key="y_slider")
        st.session_state.manual_y = manual_y
        st.write(f"`{manual_y} px`")
        
        # Ширина
        st.write("**📏 Ширина рамки**")
        manual_width = st.slider("Ширина", 50, w, manual_width, 3, key="w_slider")
        st.session_state.manual_width = manual_width
        st.write(f"`{manual_width} px`")
        
        st.divider()
        
        # Сброс
        if st.button("🔄 Сбросить", use_container_width=True):
            st.session_state.manual_x = int(w * 0.7)
            st.session_state.manual_y = h // 2
            st.session_state.manual_width = int(w * 0.2)
            st.rerun()
        
        st.divider()
        
        # ЗУМ
        st.subheader("🔍 Масштаб")
        zoom = st.slider("Зум", 300, 1200, zoom, 50, key="zoom_slider")
        st.session_state.zoom = zoom
        st.write(f"`{zoom} px`")
    
    with col_right:
        # Расчёт PD
        card_width_px = manual_width
        mm_per_px = CARD_WIDTH_MM = 85.60
        pd_mm = pd_px * mm_per_px / card_width_px
        
        # 🔥 PD крупно
        st.subheader("📏 Результат")
        col_pd1, col_pd2 = st.columns(2)
        with col_pd1:
            st.metric("PD", f"{pd_mm:.1f} мм")
        with col_pd2:
            if 58 <= pd_mm <= 72:
                st.success("✅ Норма")
            elif pd_mm < 55 or pd_mm > 75:
                st.error("⚠️ Проверь")
            else:
                st.warning("⚠️ Граница")
        
        # 🔥 Рамка
        CARD_ASPECT_RATIO = 1.586
        demo_w = int(card_width_px)
        demo_h = int(card_width_px / CARD_ASPECT_RATIO)
        box_demo = np.array([
            [manual_x, manual_y],
            [manual_x + demo_w, manual_y],
            [manual_x + demo_w, manual_y + demo_h],
            [manual_x, manual_y + demo_h]
        ], dtype=np.int32)
        
        img_debug = img_bgr.copy()
        cv2.drawContours(img_debug, [box_demo], -1, (255, 165, 0), 1)
        cv2.putText(img_debug, f"PD: {pd_mm:.1f} mm", (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 2)
        cv2.circle(img_debug, (manual_x, manual_y), 5, (0, 255, 0), -1)
        
        # 🔥 Зрачки
        cv2.circle(img_debug, (left_x, left_y), 5, (255, 0, 0), -1)
        cv2.circle(img_debug, (right_x, right_y), 5, (255, 0, 0), -1)
        cv2.line(img_debug, (left_x, left_y), (right_x, right_y), (255, 0, 0), 2)
        
        img_debug_rgb = cv2.cvtColor(img_debug, cv2.COLOR_BGR2RGB)
        
        # 🔥 Фото
        st.image(img_debug_rgb, caption=f"Результат ({zoom}px)", width=zoom)
        
        # 🔥 Скачать
        buf = BytesIO()
        Image.fromarray(img_debug_rgb).save(buf, format="PNG")
        st.download_button(
            label="💾 Скачать",
            data=buf.getvalue(),
            file_name="pd_result.png",
            mime="image/png",
            use_container_width=True
        )
    
    with st.expander("❓ Как пользоваться"):
        st.write("""
        1. **Зелёная точка** = левый верхний угол рамки
        2. **Ползунки** — настройка с шагом 3 px (точно!)
        3. **🔍 Зум** — увеличить фото для проверки
        4. **Сброс** — вернуть рамку в исходное
        5. Норма PD: **58-72 мм**
        """)