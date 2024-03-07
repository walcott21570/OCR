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
from math import sin, cos, radians
import statistics

import sys
import pathlib

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'c:\Program Files\Tesseract-OCR\tesseract.exe'


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

def process_image(file_path, page):
    if file_path.lower().endswith('.pdf'):
        images = convert_from_path(file_path, poppler_path=r"c:\poppler-23.11.0\Library\bin")
        image = cv2.cvtColor(np.array(images[page]), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(file_path)
    angle = get_rotation_angle(image)
    rotated = rotate_image(image, angle)
    return rotated

def results(rotated):
    ocr_data = pytesseract.image_to_data(rotated, lang='rus+eng', output_type=pytesseract.Output.DICT)
    # Фильтрация и вычисление средней уверенности для исходного изображения
    confidences = [conf for conf in ocr_data['conf']]
    original_avg_conf = round(statistics.mean(confidences), 2)
    if original_avg_conf<50:
        rotated_image = cv2.rotate(rotated, cv2.ROTATE_180)
        ocr_data_rotated = pytesseract.image_to_data(rotated_image, lang='rus+eng', output_type=pytesseract.Output.DICT)
        confidences_rotated = [conf for conf in ocr_data_rotated['conf']]
        rotated_avg_conf = round(statistics.mean(confidences_rotated), 2)
        if rotated_avg_conf > original_avg_conf:
            print(f'Изображение было повернуто. Средняя уверенность OCR: {rotated_avg_conf}')
            return rotated_image, rotated_avg_conf
        else:
            print(f'Изображение оставлено без изменений. Средняя уверенность OCR: {original_avg_conf} > {rotated_avg_conf}')
            return rotated, original_avg_conf
        
    else:
        print(f'Изображение оставлено без изменений. Средняя уверенность OCR: {original_avg_conf}')
        return rotated, original_avg_conf 
        

rotated = process_image('9585739543_21_24.10.2023_9696552197.pdf', page=0)
image_haha, avg = results(rotated)
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image_haha, cv2.COLOR_BGR2RGB))
plt.show()