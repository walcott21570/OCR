import cv2
import matplotlib.pyplot as plt
import pytesseract
import numpy as np
import pandas as pd
from pdf2image import convert_from_path

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

image = process_image('data/Акт № 1_08022_1014 от 30.09.23.PDF')

# remove color info
gray_image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# (1) thresholding image
ret,thresh_value = cv2.threshold(gray_image,180,255,cv2.THRESH_BINARY_INV)

# (2) dilating image to glue letter with e/a
kernel = np.ones((2,2),np.uint8)    
dilated_value = cv2.dilate(thresh_value,kernel,iterations = 1)

# (3) looking for countours
contours, hierarchy = cv2.findContours(dilated_value,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# (4) extracting coordinates and filtering them empirically 
coordinates = []
for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    if h>20 and w>30 and h*w<350000:  
        coordinates.append((x,y,w,h))

def sort2(val):   #helper for sorting by y
    return val[1]   

recognized_table = []
prev_y = 0
coordinates.sort() #sort by x
coordinates.sort(key = sort2) # sort by y
row = []

for coord in coordinates:
    x, y, w, h = coord
    # Начало новой строки, если значение y изменилось
    if y > prev_y + 5 and row:
        recognized_table.append([cell["text"] for cell in row])
        row = []
    # Обрезка изображения по текущему bbox
    crop_img = image[y:y+h, x:x+w]
    # Использование Tesseract для распознавания текста на обрезанном изображении
    recognized_string = pytesseract.image_to_string(crop_img, lang="rus").replace("\n"," ")
    # Добавление обрезанного изображения и распознанного текста в текущую строку
    #plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    #plt.show()
    #print(recognized_string)
    row.append({
        "image": crop_img,
        "text": recognized_string
    })
    prev_y = y

# Добавление последней строки, если она не пуста
if row:
    recognized_table.append([cell["text"] for cell in row])

#print(recognized_table)
df = pd.DataFrame(recognized_table)
print(df)
