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
        images = convert_from_path(file_path, poppler_path=r"g:\vadim\poppler-24.02.0\Library\bin")
        image = cv2.cvtColor(np.array(images[1]), cv2.COLOR_RGB2BGR)  # обработка первой страницы
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
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    plt.show()

    return rotated

rotated = process_image('data/image_1.jpg')
plt.imshow(rotated)

def text_detection(rotated):
    reader = easyocr.Reader(['ru'])  
    horizontal_list, _ = reader.detect(rotated)  
    return horizontal_list[0]

def make_prediction_easyocr(rotated):
    reader = easyocr.Reader(['ru'])  
    result  = reader.readtext(rotated) 
    return result

def rotated_image_check(rotated, result):
    confidences_original = []
    for (bbox, text, confidence) in result:
        confidences_original.append(confidence)
    original_avg_conf = round(statistics.mean(confidences_original), 2)
    print(original_avg_conf)
    if original_avg_conf<0.5:
        confidences_change = []   
        rotated_change = cv2.rotate(rotated, cv2.ROTATE_180)
        reader = easyocr.Reader(['ru'])  
        result_change  = reader.readtext(rotated_change)
        for (bbox, text, confidence) in result_change:
            confidences_change.append(confidence)
        change_avg_conf = round(statistics.mean(confidences_change), 2)
        print(change_avg_conf)

        if original_avg_conf > change_avg_conf:
            print(f'Изображение оставлено без изменений. Средняя уверенность первых значений OCR: {original_avg_conf} > {change_avg_conf}')
            return rotated, result
        else:
            print(f'Изображение было повернуто. Средняя уверенность первых значений OCR: {change_avg_conf} > {original_avg_conf}')
            return rotated_change, result_change
    
    else:
        print(f'Изображение оставлено без изменений. Средняя уверенность OCR: {original_avg_conf}')
        return rotated, result 


result = make_prediction_easyocr(rotated)
rotated, result = rotated_image_check(rotated, result)



def rotated_image_check_abaa(rotated):
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