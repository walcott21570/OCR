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
import os
import shutil
from PIL import Image, ImageDraw, ImageFont
import torch

import sys
import pathlib

# Путь к директории 'src'
src_dir = str(pathlib.Path(__file__).parent.resolve()) + '/ocr_transformer/src'
sys.path.append(src_dir)

from const import DIR, PATH_TEST_DIR, PATH_TEST_LABELS, WEIGHTS_PATH, PREDICT_PATH
from config import MODEL, ALPHABET, N_HEADS, ENC_LAYERS, DEC_LAYERS, DEVICE, HIDDEN
from utils import prediction


import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'c:\Program Files\Tesseract-OCR\tesseract.exe'

def get_rotation_angle(image):
    # Преобразование в градации серого и применение Canny edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Применение Hough Transform для нахождения линий
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    # Вычисление углов каждой линии
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
        angles.append(angle)

    # Вычисление среднего угла наклона
    median_angle = np.median(angles)
    return median_angle

def process_image(image):
    # Получение угла наклона
    # image = cv2.cvtColor(np.array(images[2]), cv2.COLOR_RGB2BGR) 
    angle = get_rotation_angle(image)

    # Поворот изображения
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

# Streamlit интерфейс
st.title("OCR Преобразователь")

uploaded_file = st.file_uploader("Загрузите изображение или PDF", type=["png", "jpg", "jpeg", "pdf"])
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            tmpfile_path = tmpfile.name

        # Конвертация всех страниц PDF в изображения
        images = convert_from_path(tmpfile_path, poppler_path=r"c:\poppler-23.11.0\Library\bin")
        os.unlink(tmpfile_path)

        # Создание горизонтального коллажа
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        collage = Image.new('RGB', (total_width, max_height))
        
        x_offset = 0
        for img in images:
            collage.paste(img, (x_offset, 0))
            x_offset += img.size[0]

        st.image(collage, caption='Все страницы PDF')

        # Выбор страницы и её обработка
        page_number = st.selectbox('Выберите страницу для обработки', range(1, len(images) + 1)) - 1
        selected_image = np.array(images[page_number].convert('RGB'))
        rotated_brg = cv2.cvtColor(selected_image, cv2.COLOR_RGB2BGR)  # Конвертация в BGR

        if st.button('Обработать страницу'):
            rotated = process_image(rotated_brg)
            rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)  # Конвертация обратно в RGB для отображения
            st.image(rotated_rgb, caption='Обработанное изображение')

    else:
        selected_image = np.array(Image.open(uploaded_file).convert('RGB'))
        rotated = process_image(selected_image)
        st.image(rotated, caption='Обработанное изображение')


reader = easyocr.Reader(['ru'])
horizontal_list, _  = reader.detect(rotated)

maximum_y = rotated.shape[0]
maximum_x = rotated.shape[1]
rotated_viz = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

for box in horizontal_list[0]:
    x_min = max(0,box[0])
    x_max = min(box[1],maximum_x)
    y_min = max(0,box[2])
    y_max = min(box[3],maximum_y)
    cv2.rectangle(rotated_viz, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

# Отображение изображения с нарисованными прямоугольниками
st.image(rotated_viz, caption='Обнаружение текста')

data = []

# Извлекаем текст и уровень уверенности для каждого bbox
for bbox in horizontal_list[0]:
    # Вырезаем область изображения по bbox
    x_min, x_max, y_min, y_max = bbox
    crop_img = rotated[y_min:y_max, x_min:x_max]

    # Распознавание текста в этой области
    text = pytesseract.image_to_string(crop_img, lang='rus').strip()  # Используйте 'rus' для русского языка
    
    # Получаем уровень уверенности для распознанного текста
    ocr_data = pytesseract.image_to_data(crop_img, lang='rus', output_type=pytesseract.Output.DICT)
    try:
        # Берем первое значение confidence, которое не равно -1
        confidences = [conf for conf, text in zip(ocr_data['conf'], ocr_data['text']) if conf != -1 and text.strip()]
        confidence = confidences[0] / 100.0 if confidences else -1
    except IndexError:
        confidence = -1

    # Добавляем данные в список
    data.append({'bbox': bbox, 'text': text, 'confidence': confidence})

# Создаем DataFrame из списка
df = pd.DataFrame(data)

# Загрузите исходное изображение и конвертируйте его в формат PIL
# rotated = cv2.imread('путь_к_вашему_изображению')
pil_image = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

# Путь к папке 'test'
test_folder_path = 'test'

# Удаление папки 'test', если она существует
if os.path.exists(test_folder_path):
    shutil.rmtree(test_folder_path)

# Создание папки 'test'
os.makedirs(test_folder_path)

# Перебор всех строк DataFrame
for index, row in df.iterrows():
    # Проверка уровня уверенности
    if row['confidence'] < 0.5:
        # Получение координат bbox
        x_min, x_max, y_min, y_max = row['bbox']

        # Корректировка координат bbox, если это необходимо
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
        y_min, y_max = min(y_min, y_max), max(y_min, y_max)

        # Обрезка изображения по bbox
        cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))

        # Сохранение изображения в папку 'test'
        file_path = os.path.join('test', f'image_{index}.png')
        cropped_image.save(file_path)
        
# Создание словаря символов
char2idx = {char: idx for idx, char in enumerate(ALPHABET)}
idx2char = {idx: char for idx, char in enumerate(ALPHABET)}

# Создание модели
if MODEL == 'model1':
    from models import model1
    model = model1.TransformerModel(len(ALPHABET), hidden=HIDDEN, enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,   
                                    nhead=N_HEADS, dropout=0.0).to(DEVICE)
elif MODEL == 'model2':
    from models import model2
    model = model2.TransformerModel(len(ALPHABET), hidden=HIDDEN, enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,   
                                    nhead=N_HEADS, dropout=0.0).to(DEVICE)

# Загрузка весов модели
if WEIGHTS_PATH is not None:
    print(f'loading weights from {WEIGHTS_PATH}')
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
    
def make_predictions(model, predict_path, char2idx, idx2char):
    preds = prediction(model, predict_path, char2idx, idx2char)

    # Сохранение результатов предсказаний
    with open(DIR / 'predictions.tsv', 'w', encoding='utf-8') as f:
        f.write('filename\tprediction\n')
        for item in preds.items():
            f.write(f"{item[0]}\t{item[1]}\n")
            
    print(f'Predictions are saved in {DIR}/predictions.tsv')

# Функция для выполнения предсказаний
make_predictions(model, PREDICT_PATH, char2idx, idx2char)

# Чтение файла TSV
predictions_df = pd.read_csv('predictions.tsv', sep='\t')

# Преобразование имени файла в индекс
predictions_df['index'] = predictions_df['filename'].str.extract(r'image_(\d+).png').astype(int)

# Инициализация нового столбца для предсказаний второй модели как столбца строк
df['second_model_prediction'] = None  # или использовать '' для пустой строки

# Обновление значений в df на основе предсказаний второй модели
for idx, row in predictions_df.iterrows():
    df_index = row['index']
    df.at[df_index, 'second_model_prediction'] = row['prediction']
    
# Создание нового DataFrame с измененными значениями в столбце text
new_df = df.copy()

# Функция для замены текста
def replace_text(row):
    if row['confidence'] < 0.5:
        return row['second_model_prediction']
    else:
        return row['text']

# Применение функции к каждой строке
new_df['text'] = new_df.apply(replace_text, axis=1)
new_df = new_df.drop('second_model_prediction', axis=1)

# Загрузка исходного изображения
# rotated = cv2.imread('путь_к_вашему_изображению')
pil_original_image = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

# Получение размеров исходного изображения
height, width, _ = rotated.shape

# Создание белого изображения тех же размеров для текста
blank_image = Image.new('RGB', (width, height), (255, 255, 255))
draw = ImageDraw.Draw(blank_image)

# Настройки шрифта для текста
font_path = 'DejaVuSans.ttf'  # Укажите путь к файлу шрифта
font_size = 20
font = ImageFont.truetype(font_path, font_size)

# Рисование текста на белом изображении в соответствии с bbox
for _, row in new_df.iterrows():
    x_min, x_max, y_min, y_max = row['bbox']
    text = row['text']
    
    # Расчет позиции для текста
    text_position = (x_min, y_max - font_size)

    # Рисование текста
    draw.text(text_position, text, fill=(0, 0, 0), font=font)

# Создание нового изображения, которое будет включать оба изображения (исходное и с текстом)
total_width = width * 2
combined_image = Image.new('RGB', (total_width, height))

# Размещение исходного и текстового изображений на общей канве
combined_image.paste(pil_original_image, (0, 0))
combined_image.paste(blank_image, (width, 0))

st.image(combined_image, caption='Распознование текста')