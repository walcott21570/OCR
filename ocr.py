import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from pdf2image import convert_from_path
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

def process_image(file_path):
    # Определение формата файла
    if file_path.lower().endswith('.pdf'):
        # Обработка PDF
        images = convert_from_path(file_path, poppler_path=r"c:\poppler-23.11.0\Library\bin")
        image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)  # обработка первой страницы
    else:
        # Обработка изображений в формате JPG
        image = cv2.imread(file_path)
   
   # Получение угла наклона
    angle = get_rotation_angle(image)

    # Поворот изображения
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Вывод или сохранение результатов
    ##plt.figure(figsize=(10, 10))
    #plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    #plt.show()

    return rotated

rotated = process_image('data/Акт № 1_08022_1014 от 30.09.23.PDF')
print(rotated)

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


# Загружаем исходное изображение и конвертируйте его в формат PIL
pil_image = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

def make_predictions(model, images_dict, char2idx, idx2char):
    preds = {}
    for index, img in images_dict.items():
        prediction_result = prediction(model, img, char2idx, idx2char)
        preds[index] = prediction_result

    return preds


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

# Подготовка списка изображений для предсказания
images_to_predict = {}

for index, row in df.iterrows():
    if row['confidence'] < 0.5:
        x_min, x_max, y_min, y_max = row['bbox']
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
        y_min, y_max = min(y_min, y_max), max(y_min, y_max)

        cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))
        images_to_predict[index] = cropped_image

# Выполнение предсказаний
pred = make_predictions(model, images_to_predict, char2idx, idx2char)

for index, prediction in pred.items():
    df.at[index, 'text'] = prediction

new_df = df.copy()

# Загрузка исходного изображения
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

print(list(new_df['text'].values))
# Отображение результата
plt.figure(figsize=(15, 15))
plt.imshow(combined_image)
plt.axis('off')
plt.show()