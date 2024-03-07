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
import time

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

    return rotated

rotated = process_image('data/Акт № 1_08022_1014 от 30.09.23.PDF')
#print(rotated)

# Инициализация первого детектора (craft)
start_time_craft = time.time()  # Начало замера времени для craft
reader_craft = easyocr.Reader(['ru', 'en'], gpu=False, detect_network='craft')
horizontal_list_craft, _ = reader_craft.detect(rotated)
end_time_craft = time.time()  # Конец замера времени для craft
execution_time_craft = end_time_craft - start_time_craft  # Расчет времени выполнения для craft

# Инициализация второго детектора (dbnet18)
start_time_dbnet = time.time()  # Начало замера времени для dbnet18
reader_dbnet = easyocr.Reader(['ru', 'en'], gpu=False, detect_network='dbnet18')
horizontal_list_dbnet, _ = reader_dbnet.detect(rotated)
end_time_dbnet = time.time()  # Конец замера времени для dbnet18
execution_time_dbnet = end_time_dbnet - start_time_dbnet  # Расчет времени выполнения для dbnet18

print(horizontal_list_craft)
print(horizontal_list_dbnet)
# Вывод результатов времени выполнения
print(f"Время выполнения для детектора craft: {execution_time_craft:.2f} секунд")
print(f"Время выполнения для детектора dbnet18: {execution_time_dbnet:.2f} секунд")

# Подсчет и сравнение количества обнаруженных текстовых блоков
print(f"Количество обнаруженных текстовых блоков детектором craft: {len(horizontal_list_craft[0])}")
print(f"Количество обнаруженных текстовых блоков детектором dbnet18: {len(horizontal_list_dbnet[0])}")

# Функция для визуализации результатов детекции
def visualize_detection(rotated_img, horizontal_list, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
    for box in horizontal_list[0]:
        x_min, x_max = max(0, box[0]), min(box[1], rotated_img.shape[1])
        y_min, y_max = max(0, box[2]), min(box[3], rotated_img.shape[0])
        plt.gca().add_patch(patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none'))
    plt.title(title)
    plt.show()

# Визуализация результатов для каждого детектора
visualize_detection(rotated, horizontal_list_craft, "Результаты детектора Craft")
visualize_detection(rotated, horizontal_list_dbnet, "Результаты детектора DBNet18")

'''
maximum_y = rotated.shape[0]
maximum_x = rotated.shape[1]
rotated_viz = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)


for box in horizontal_list[0]:
    x_min = max(0,box[0])
    x_max = min(box[1],maximum_x)
    y_min = max(0,box[2])
    y_max = min(box[3],maximum_y)
    cv2.rectangle(rotated_viz, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

plt.imshow(rotated_viz)
plt.show()
'''