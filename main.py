import streamlit as st
from PIL import Image
from tensorflow.keras.utils import img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import matplotlib.pyplot as plt

# Fungsi untuk memuat model dengan caching
@st.cache_resource
def load_model_cached():
    return load_model('FV.h5')

# Load model pra-terlatih
model = load_model_cached()

# Label untuk prediksi model
labels = {0: 'apel', 1: 'pisang', 2: 'bit', 3: 'paprika', 4: 'kubis', 5: 'cabai', 6: 'wortel',
            7: 'kembang kol', 8: 'cabai', 9: 'jagung', 10: 'mentimun', 11: 'terong', 12: 'bawang putih', 13: 'jahe',
            14: 'anggur', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'selada', 19: 'mangga', 20: 'bawang merah', 21: 'jeruk',
            22: 'paprika', 23: 'pir', 24: 'kacang polong', 25: 'nanas', 26: 'delima', 27: 'kentang', 28: 'lobak',
            29: 'kacang kedelai', 30: 'bayam', 31: 'jagung manis', 32: 'ubi jalar', 33: 'tomat', 34: 'turnip',
            35: 'semangka'}

# Daftar buah-buahan dan sayuran
buah = ['apel', 'pisang', 'paprika', 'cabai', 'anggur', 'jalepeno', 'kiwi', 'lemon', 'mangga', 'jeruk', 'paprika', 'pir', 'nanas', 'delima', 'semangka']
sayuran = ['bit', 'kubis', 'cabai', 'wortel', 'kembang kol', 'jagung', 'mentimun', 'terong', 'jahe',
                'selada', 'bawang merah', 'kacang polong', 'kentang', 'lobak', 'kacang kedelai', 'bayam', 'jagung manis', 'ubi jalar',
                'tomat', 'turnip']

# Fungsi untuk mendapatkan informasi kalori dari Google
def fetch_calories(prediction):
    try:
        url = f'https://www.google.com/search?&q=kalori dalam {prediction}'
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        
        # Mencoba berbagai strategi untuk menemukan informasi kalori
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd")
        if not calories:
            calories = scrap.find("div", class_="BNeawe tAd8D AP7Wnd")
        if not calories:
            calories = scrap.find("span", class_="BNeawe iBp4i AP7Wnd")
        if not calories:
            calories = scrap.find("div", {"role": "heading"})
        
        if calories:
            cal_text = calories.text
            # Memeriksa teks non-numerik atau generik
            if "Tampilkan semua" in cal_text or not any(char.isdigit() for char in cal_text):
                return "Informasi kalori tidak ditemukan."
            return cal_text
        else:
            return "Informasi kalori tidak ditemukan."
    except Exception as e:
        st.error("Tidak bisa mengambil informasi kalori.")
        print(e)
        return None

# Fungsi untuk memproses gambar dan membuat prediksi
def processed_img(image_pil):
    img = img_to_array(image_pil.resize((224, 224)).convert('RGB'))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int(y_class[0])
    res = labels[y]
    return res.capitalize()

# Fungsi untuk memproses frame dari webcam dan membuat prediksi
def process_frame(frame, threshold):  
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((224, 224))
    img = img_to_array(image_pil)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    answer = model.predict(img)
    confidence = np.max(answer)
    if confidence > threshold:
        y_class = answer.argmax(axis=-1)
        y = int(y_class[0])
        res = labels[y]
        return res.capitalize(), confidence
    else:
        return None, confidence

# Fungsi untuk menemukan bounding box dari kontur terbesar
def find_largest_contour_bounding_box(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h, cv2.contourArea(largest_contour)
    return None, None, None, None, 0

# Fungsi untuk memotong gambar agar sesuai dengan tepi buah/sayuran menggunakan Canny Edge Detection
def crop_to_fit(image):
    open_cv_image = np.array(image.convert("RGB"))

    mask = np.zeros(open_cv_image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, open_cv_image.shape[1] - 20, open_cv_image.shape[0] - 20)
    cv2.grabCut(open_cv_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented = open_cv_image * mask2[:, :, np.newaxis]

    b, g, r = cv2.split(segmented)
    alpha = np.ones(b.shape, dtype=b.dtype) * 255
    alpha[mask2 == 0] = 0
    rgba = cv2.merge([b, g, r, alpha])

    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        crop = rgba[y:y+h, x:x+w]
    else:
        crop = rgba

    return Image.fromarray(crop, 'RGBA')

# Fungsi transformasi Fourier
def transformasi_fourier(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    shift = np.fft.fftshift(dft)
    row, col = image.shape
    center_row, center_col = row // 2, col // 2

    mask = np.zeros((row, col, 2), np.uint8)
    mask[center_row - 30:center_row + 30, center_col - 30:center_col + 30] = 1

    fshift = shift * mask

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

    return img_back

# Fungsi visualisasi transformasi Fourier
def visualisasi(image, imageThen):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    ax[1].imshow(imageThen, cmap='gray')
    ax[1].set_title('Magnitude Spectrum')
    ax[1].axis('off')

    st.pyplot(fig)

st.title("Klasifikasi Buah 🍍-Sayuran 🍅")

st.markdown(
    """
    <div style="padding: 10px; border-radius: 5px; font-size: 12px;">
        <strong>Tugas Besar 2 Pengolahan Citra</strong><br>
        Disusun oleh:<br>
        1. Agil Dwiki Yudistira (41522110068)<br>
        2. Sebastianus Lukito (41522110051)<br>
        3. Ridho Pangestu (41522110041)<br>
        Dosen Pengampu: Nur Ismawati, S.T., M.Cs.
    </div>
    """, unsafe_allow_html=True
)

# Kelas untuk transformasi video dengan streamlit-webrtc
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.threshold = 0.95

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        bbox_x, bbox_y, bbox_w, bbox_h, area = find_largest_contour_bounding_box(img)
        if bbox_x is not None and area > 5000:
            cropped_frame = img[bbox_y:bbox_y + bbox_h, bbox_x:bbox_x + bbox_w]
            result, confidence = process_frame(cropped_frame, self.threshold)
            if result and (result.lower() in buah or result.lower() in sayuran):
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                color = (0, 255, 0)
                thickness = 2
                text = f"Prediksi: {result} ({confidence*100:.2f}%)"
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(img, (10, 30), (10 + text_width, 30 - text_height - baseline), color, thickness=cv2.FILLED)
                cv2.putText(img, text, (10, 30 - baseline), font, font_scale, (0, 0, 0), thickness)
                cv2.rectangle(img, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 0), 2)
        return img

def main():
    option = st.selectbox('Pilih metode input:', ('Unggah Gambar', 'Gunakan Kamera'))
    
    if option == 'Unggah Gambar':
        img_file = st.file_uploader("Pilih Gambar", type=["jpg", "png"])
        if img_file is not None:
            image_pil = Image.open(img_file)
            cropped_image = crop_to_fit(image_pil)
            width, height = cropped_image.size
            new_height = 250
            new_width = int((new_height / height) * width)
            img = cropped_image.resize((new_width, new_height))
            st.image(img, use_column_width=False)
            result = processed_img(cropped_image)
            if result.lower() in labels.values():
                st.info(f'**Kategori: {"Sayuran" if result.lower() in sayuran else "Buah"}**')
            st.success(f"**Prediksi: {result}**")
            cal = fetch_calories(result)
            if cal:
                st.warning(f'**{cal} (per 100 gram)**')
                
            # Transformasi Fourier dan visualisasi
            image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
            fourier_image = transformasi_fourier(image_cv)
            visualisasi(image_cv, fourier_image)
    
    elif option == 'Gunakan Kamera':
        webrtc_streamer(key="sample", video_transformer_factory=VideoTransformer)

if __name__ == '__main__':
    main()
