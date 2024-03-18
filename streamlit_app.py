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

#@st.cache_data
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

#@st.cache_data
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

def make_prediction_easyocr(rotated):
    reader = easyocr.Reader(['ru'], gpu=False)  
    result  = reader.readtext(rotated) 
    return result

def rotated_image_check(rotated, result):
    confidences_original = []
    for (bbox, text, confidence) in result:
        confidences_original.append(confidence)
    original_avg_conf = round(statistics.mean(confidences_original), 2)
    if original_avg_conf<0.5:
        confidences_change = []   
        rotated_change = cv2.rotate(rotated, cv2.ROTATE_180)
        reader = easyocr.Reader(['ru'], gpu=False)  
        result_change  = reader.readtext(rotated_change)
        for (bbox, text, confidence) in result_change:
            confidences_change.append(confidence)
        change_avg_conf = round(statistics.mean(confidences_change), 2)

        if original_avg_conf > change_avg_conf:
            print(f'Изображение оставлено без изменений. Средняя уверенность первых значений OCR: {original_avg_conf} > {change_avg_conf}')
            return rotated, result
        else:
            print(f'Изображение было повернуто. Средняя уверенность первых значений OCR: {change_avg_conf} > {original_avg_conf}')
            return rotated_change, result_change
    
    else:
        print(f'Изображение оставлено без изменений. Средняя уверенность OCR: {original_avg_conf}')
        return rotated, result 

def visualize_text_detection(rotated, result):
    rotated_viz = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    
    for (bbox, text, easyocr_confidence) in result:
        # Получение координат рамки
        (x_min, y_min), (x_max, y_max) = bbox[0], bbox[2]
        
        # Проверка и корректировка координат рамки
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
        y_min, y_max = min(y_min, y_max), max(y_min, y_max)

        cv2.rectangle(rotated_viz, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

    return rotated_viz

def make_prediction_tesseract(rotated, result):
    # Создание DataFrame для сохранения результатов
    results_df = pd.DataFrame(columns=['bbox', 'easyocr_text', 'easyocr_confidence', 'tesseract_text', 'tesseract_confidence'])

    rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rotated_rgb)

    rows_list = []  # Список для временного хранения данных перед добавлением в DataFrame

    for (bbox, text, easyocr_confidence) in result:
        (x_min, y_min), (x_max, y_max) = bbox[0], bbox[2]
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
        y_min, y_max = min(y_min, y_max), max(y_min, y_max)

        cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))

        tesseract_data = pytesseract.image_to_data(cropped_image, lang='rus', output_type=pytesseract.Output.DICT)
        tesseract_texts = [t for t, conf in zip(tesseract_data['text'], tesseract_data['conf']) if int(conf) > -1 and t.strip()]
        tesseract_text = ' '.join(tesseract_texts).strip()
        tesseract_conf = tesseract_data['conf'][-1]
        tesseract_confidence = tesseract_conf / 100 if tesseract_conf != -1 else tesseract_conf

        easyocr_confidence = round(easyocr_confidence, 2)

        rows_list.append({'bbox': bbox, 'easyocr_text': text, 'easyocr_confidence': easyocr_confidence, 'tesseract_text': tesseract_text, 'tesseract_confidence': tesseract_confidence})

    # Добавление собранных данных в DataFrame одной операцией
    if rows_list:
        results_df = pd.DataFrame(rows_list)
    
    return results_df

#@st.cache_data
def create_model_transformer(model_type, alphabet, hidden, enc_layers, dec_layers, n_heads, dropout, device, weights_path):
    # Создание модели в зависимости от указанного типа
    if model_type == 'model1':
        from models import model1
        model = model1.TransformerModel(len(alphabet), hidden=hidden, enc_layers=enc_layers, dec_layers=dec_layers, nhead=n_heads, dropout=dropout).to(device)
    elif model_type == 'model2':
        from models import model2
        model = model2.TransformerModel(len(alphabet), hidden=hidden, enc_layers=enc_layers, dec_layers=dec_layers, nhead=n_heads, dropout=dropout).to(device)

    # Загрузка весов модели, если путь указан
    if weights_path is not None:
        #print(f'loading weights from {weights_path}')
        model.load_state_dict(torch.load(weights_path, map_location=device))

    return model

#@st.cache_data
def make_prediction_transforomer(_model, df, rotated, alphabet, confidence_threshold=0.5):
    preds = {}
    pil_image = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    char2idx = {char: idx for idx, char in enumerate(alphabet)}
    idx2char = {idx: char for idx, char in enumerate(alphabet)}

    for index, row in df.iterrows():
        if row['easyocr_confidence'] < confidence_threshold and row['tesseract_confidence'] < confidence_threshold:
            bbox = row['bbox']
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]

            # Находим минимальные и максимальные координаты
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))
            prediction_result = prediction(model, cropped_image, char2idx, idx2char)
            preds[index] = prediction_result

    return preds

#@st.cache_data
def create_combined_image(pred, df, rotated, font_path='DejaVuSans.ttf', font_size=20):
    for index, prediction in pred.items():
        df.at[index, 'text'] = prediction

    df.to_csv('test.csv')

    # Создаем список для хранения данных
    data = []

    for index, row in df.iterrows():
        # Проверка, если уверенность обоих OCR меньше 0.5
# Проверка, если уверенность обоих OCR меньше 0.4
        if row['easyocr_confidence'] < 0.4 and row['tesseract_confidence'] < 0.4:
            # Сохраняем текст из исходного DataFrame, если он есть
            text = row['text'] if pd.notnull(row['text']) and row['text'].strip() else ""
        elif row['tesseract_confidence'] >= 0.4:
            # Используем tesseract_text, если его уверенность >= 0.4
            text = row['tesseract_text']
        else:
            # Во всех остальных случаях предпочтем easyocr_text, если уверенность easyocr_confidence доступна
            # Если уверенность easyocr_confidence не доступна, используем tesseract_text
            text = row['easyocr_text'] if pd.notnull(row['easyocr_confidence']) else row['tesseract_text']
        
        # Выбираем наибольшую уверенность
        confidence = max(row['easyocr_confidence'], row['tesseract_confidence'])

        data.append({'bbox': row['bbox'], 'text': text, 'confidence': confidence})

    final_df = pd.DataFrame(data, columns=['bbox', 'text', 'confidence'])
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
    for _, row in final_df.iterrows():
        x_min, y_min = row['bbox'][0]
        x_max, y_max = row['bbox'][1]
        text = row['text']
        text_position = (x_min, y_max - font_size)
        draw.text(text_position, text, fill=(0, 0, 0), font=font)

    # Создание нового изображения, которое будет включать оба изображения
    total_width = width * 2
    combined_image = Image.new('RGB', (total_width, height))

    # Размещение исходного и текстового изображений
    combined_image.paste(pil_original_image, (0, 0))
    combined_image.paste(blank_image, (width, 0))

    return combined_image, final_df

st.title("OCR Преобразователь")
uploaded_file = st.file_uploader("Загрузите изображение или PDF", type=["png", "jpg", "jpeg", "pdf"])
if uploaded_file is not None:
    images = uploaded_image(uploaded_file)
    collage = prepate_collage(images)
    st.image(collage, caption='Все страницы')
    page_number = st.selectbox('Выберите страницу для обработки', range(1, len(images) + 1)) - 1
    if st.button('Обработать страницу') and page_number is not None:
        rotated = get_page(images, page_number)
        result = make_prediction_easyocr(rotated)
        rotated, result = rotated_image_check(rotated, result)  
        rotated_viz = visualize_text_detection(rotated, result)
        st.image(rotated_viz, caption='Обнаружение текста')
        df = make_prediction_tesseract(rotated, result)
        model = create_model_transformer(MODEL, ALPHABET, HIDDEN, ENC_LAYERS, DEC_LAYERS, N_HEADS, 0.0, DEVICE, WEIGHTS_PATH)
        pred = make_prediction_transforomer(model, df, rotated, ALPHABET, confidence_threshold=0.5)
        combined_image, final_df = create_combined_image(pred, df, rotated, font_path='DejaVuSans.ttf', font_size=20)
        st.image(combined_image, caption='Распознование текста')
        text_to_display = ' '.join(final_df['text'].astype(str).tolist())
        st.write(text_to_display)
        