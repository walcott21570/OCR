import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from pdf2image import convert_from_path
import tempfile
import easyocr
import pandas as pd
import subprocess
from PIL import Image
import shutil
from PIL import Image, ImageDraw, ImageFont
import torch
from math import sin, cos, radians
import statistics

import sys
import pathlib

# Путь к директории 'src'
src_dir = str(pathlib.Path(__file__).parent.resolve()) + '/ocr_transformer/src'
sys.path.append(src_dir)

from const import DIR, PATH_TEST_DIR, PATH_TEST_LABELS, WEIGHTS_PATH, PREDICT_PATH
from config import MODEL, ALPHABET, N_HEADS, ENC_LAYERS, DEC_LAYERS, DEVICE, HIDDEN
from utils import prediction


import pytesseract
#pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = r'g:\vadim\Tesseract-OCR\tesseract.exe'  # r'c:\Program Files\Tesseract-OCR\tesseract.exe'

@st.cache_data
def get_rotation_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
        angles.append(angle)
    median_angle = np.median(angles)
    return median_angle

def get_new_size(width, height, angle):
    angle = radians(angle)
    abs_cos = abs(cos(angle))
    abs_sin = abs(sin(angle))
    new_width = int((height * abs_sin) + (width * abs_cos))
    new_height = int((height * abs_cos) + (width * abs_sin))
    return new_width, new_height

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    new_width, new_height = get_new_size(w, h, angle)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_width - w) / 2
    M[1, 2] += (new_height - h) / 2
    rotated = cv2.warpAffine(image, M, (new_width, new_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

@st.cache_data
def process_image(image):
    angle = get_rotation_angle(image)
    rotated = rotate_image(image, angle)

    return rotated

#@st.cache_data
def uploaded_image(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            tmpfile_path = tmpfile.name

        images = convert_from_path(tmpfile_path, poppler_path=r"g:\vadim\poppler-24.02.0\Library\bin") # , poppler_path=r"c:\poppler-23.11.0\Library\bin"
        os.unlink(tmpfile_path)
    else:
        images = [Image.open(uploaded_file).convert('RGB')]
    return images

#@st.cache_data
def prepate_collage(_images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    collage = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for img in images:
        collage.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    return np.array(collage)

def get_page(images, page_number):
    selected_image = np.array(images[page_number].convert('RGB'))
    rotated_bgr = cv2.cvtColor(selected_image, cv2.COLOR_RGB2BGR)

    rotated = process_image(rotated_bgr)
    rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    st.image(rotated_rgb, caption='Обработанное изображение')
    return rotated

@st.cache_data
def text_detection(rotated):
    reader = easyocr.Reader(['ru'])  
    horizontal_list, _ = reader.detect(rotated)  
    return horizontal_list[0]

def rotated_image_check(rotated):
    horizontal_original = text_detection(rotated)
    confidences_original = []
    for bbox in horizontal_original[:5]:
        x_min, x_max, y_min, y_max = bbox
        crop_img = rotated[y_min:y_max, x_min:x_max]
        ocr_data = pytesseract.image_to_data(crop_img, lang='rus+eng', output_type=pytesseract.Output.DICT)
        confidence_1 = [conf for conf in ocr_data['conf']]
        confidences_original.append(confidence_1)
    original_avg_conf = round(statistics.mean([item for sublist in confidences_original for item in sublist]), 2)
    
    if original_avg_conf<50:
        confidences_change = []
        rotated_image = cv2.rotate(rotated, cv2.ROTATE_180)
        horizontal_changes = text_detection(rotated_image)
        for bbox in horizontal_changes[:5]:
            x_min, x_max, y_min, y_max = bbox
            crop_img = rotated_image[y_min:y_max, x_min:x_max]
            ocr_data = pytesseract.image_to_data(crop_img, lang='rus+eng', output_type=pytesseract.Output.DICT)
            confidence_2 = [conf for conf in ocr_data['conf']]
            confidences_change.append(confidence_2)
        change_avg_conf = round(statistics.mean([item for sublist in confidences_change for item in sublist]), 2)

        if original_avg_conf > change_avg_conf:
            print(f'Изображение оставлено без изменений. Средняя уверенность первых значений OCR: {original_avg_conf} > {change_avg_conf}')
            return rotated, horizontal_original
        else:
            print(f'Изображение было повернуто. Средняя уверенность первых значений OCR: {change_avg_conf} > {original_avg_conf}')
            return rotated_image, horizontal_changes
    
    else:
        print(f'Изображение оставлено без изменений. Средняя уверенность OCR: {original_avg_conf}')
        return rotated, original_avg_conf 
    
@st.cache_data
def visualize_text_detection(rotated, horizontal_list):
    maximum_y, maximum_x = rotated.shape[:2]
    rotated_viz = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    
    for box in horizontal_list:
        x_min = max(0, int(box[0]))
        x_max = min(int(box[1]), maximum_x)
        y_min = max(0, int(box[2]))
        y_max = min(int(box[3]), maximum_y)
        cv2.rectangle(rotated_viz, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    return rotated_viz
 
@st.cache_data
def make_prediction_tesseract(rotated, horizontal_list):
    data = []

    for bbox in horizontal_list:
        x_min, x_max, y_min, y_max = bbox
        crop_img = rotated[y_min:y_max, x_min:x_max]

        text = pytesseract.image_to_string(crop_img, lang='rus+eng').strip()
        
        ocr_data = pytesseract.image_to_data(crop_img, lang='rus+eng', output_type=pytesseract.Output.DICT)
        try:
            confidences = [conf for conf, text in zip(ocr_data['conf'], ocr_data['text']) if conf != -1 and text.strip()]
            confidence = confidences[0] / 100.0 if confidences else -1
        except IndexError:
            confidence = -1

        data.append({'bbox': bbox, 'text': text, 'confidence': confidence})

    return pd.DataFrame(data)

@st.cache_data
def create_model(model_type, alphabet, hidden, enc_layers, dec_layers, n_heads, dropout, device, weights_path):
    # Создание модели в зависимости от указанного типа
    if model_type == 'model1':
        from models import model1
        model = model1.TransformerModel(len(alphabet), hidden=hidden, enc_layers=enc_layers, dec_layers=dec_layers, nhead=n_heads, dropout=dropout).to(device)
    elif model_type == 'model2':
        from models import model2
        model = model2.TransformerModel(len(alphabet), hidden=hidden, enc_layers=enc_layers, dec_layers=dec_layers, nhead=n_heads, dropout=dropout).to(device)

    # Загрузка весов модели, если путь указан
    if weights_path is not None:
        print(f'loading weights from {weights_path}')
        model.load_state_dict(torch.load(weights_path, map_location=device))

    return model

@st.cache_data
def make_prediction_transforomer(_model, df, rotated, alphabet, confidence_threshold=0.5):
    preds = {}
    pil_image = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    char2idx = {char: idx for idx, char in enumerate(alphabet)}
    idx2char = {idx: char for idx, char in enumerate(alphabet)}

    for index, row in df.iterrows():
        if row['confidence'] < confidence_threshold:
            x_min, x_max, y_min, y_max = row['bbox']
            x_min, x_max = min(x_min, x_max), max(x_min, x_max)
            y_min, y_max = min(y_min, y_max), max(y_min, y_max)

            cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))
            prediction_result = prediction(model, cropped_image, char2idx, idx2char)
            preds[index] = prediction_result

    return preds

@st.cache_data
def create_combined_image(pred, df, rotated, font_path='DejaVuSans.ttf', font_size=20):
    for index, prediction in pred.items():
        df.at[index, 'text'] = prediction
    # Загрузка исходного изображения
    pil_original_image = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

    # Получение размеров исходного изображения
    height, width, _ = rotated.shape

    # Создание белого изображения тех же размеров для текста
    blank_image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(blank_image)

    # Настройки шрифта для текста
    font = ImageFont.truetype(font_path, font_size)

    # Рисование текста на белом изображении в соответствии с bbox
    for _, row in df.iterrows():
        x_min, x_max, y_min, y_max = row['bbox']
        text = row['text']
        text_position = (x_min, y_max - font_size)
        draw.text(text_position, text, fill=(0, 0, 0), font=font)

    # Создание нового изображения, которое будет включать оба изображения
    total_width = width * 2
    combined_image = Image.new('RGB', (total_width, height))

    # Размещение исходного и текстового изображений
    combined_image.paste(pil_original_image, (0, 0))
    combined_image.paste(blank_image, (width, 0))

    return combined_image

st.title("OCR Преобразователь")
uploaded_file = st.file_uploader("Загрузите изображение или PDF", type=["png", "jpg", "jpeg", "pdf"])
if uploaded_file is not None:
    images = uploaded_image(uploaded_file)
    collage = prepate_collage(images)
    st.image(collage, caption='Все страницы')
    page_number = st.selectbox('Выберите страницу для обработки', range(1, len(images) + 1)) - 1
    if st.button('Обработать страницу') and page_number is not None:
        rotated = get_page(images, page_number)
        rotated, horizontal_list = rotated_image_check(rotated)
        rotated_viz = visualize_text_detection(rotated, horizontal_list)
        st.image(rotated_viz, caption='Обнаружение текста')
        df = make_prediction_tesseract(rotated, horizontal_list)
        model = create_model(MODEL, ALPHABET, HIDDEN, ENC_LAYERS, DEC_LAYERS, N_HEADS, 0.0, DEVICE, WEIGHTS_PATH)
        pred = make_prediction_transforomer(model, df, rotated, ALPHABET, confidence_threshold=0.5)
        combined_image = create_combined_image(pred, df, rotated, font_path='DejaVuSans.ttf', font_size=20)
        st.image(combined_image, caption='Распознование текста')
        text_to_display = ' '.join(df['text'].astype(str).tolist())
        st.write(text_to_display)
