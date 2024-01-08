import cv2
import numpy as np
from matplotlib import pyplot as plt
from pdf2image import convert_from_path
import easyocr

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
        image = cv2.cvtColor(np.array(images[2]), cv2.COLOR_RGB2BGR)  # обработка первой страницы
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
    #plt.figure(figsize=(10, 10))
    #plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    #plt.show()

    return rotated

rotated = process_image('data/test_2.jpg')
print(rotated.shape)

reader = easyocr.Reader(['ru'])
print(reader)

result  = reader.readtext(rotated)
print(result)


for box in horizontal_list:
    x_min = max(0,box[0])
    x_max = min(box[1],maximum_x)
    y_min = max(0,box[2])
    y_max = min(box[3],maximum_y)
    crop_img = img[y_min : y_max, x_min:x_max]
    width = x_max - x_min
    height = y_max - y_min
    ratio = calculate_ratio(width,height)
    new_width = int(model_height*ratio)