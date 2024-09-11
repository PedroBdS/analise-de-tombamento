import numpy as np
import cv2
import sys

VIDEO = 'C:/Users/PBOTTARI/Documents/Python/Nova pasta/teste5.mp4'

algorithm_types = ['KNN', 'GMG', 'CNT', 'MOG', 'MOG2']
a = 3
algorithm_type = algorithm_types[a]

def Kernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if KERNEL_TYPE == 'opening':
        kernel = np.ones((3, 3), np.uint8)
    if KERNEL_TYPE == 'closing':
        kernel = np.ones((3, 3), np.uint8)
    return kernel

def Filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
    if filter == 'dilation':
        return cv2.dilate(img, Kernel('dilation'), iterations=2)
    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
        dilation = cv2.dilate(opening, Kernel('dilation'), iterations=2)
        return dilation

def Subtractor(algorithm_type):
    if algorithm_type == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if algorithm_type == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if algorithm_type == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    if algorithm_type == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if algorithm_type == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    print('Detector inválido')
    sys.exit(1)

# -------------------------------------------------------------------------------------------------------------------------

linha_ROI = 150  # Posição da linha de contagem
latas = 0
latas_tombadas = 0  # Contador de latas tombadas

# Variáveis para rastreamento
detected_cans = []
memory = {}  # Memória para objetos rastreados (evita contagem dupla)
max_memory_frames = 10  # Máximo de frames para lembrar uma lata detectada
min_dist = 50  # Distância mínima entre detecções consecutivas da mesma lata

buffer = []  # Buffer para validar detecções ao longo de frames
buffer_size = 2  # Tamanho do buffer de frames consecutivos para validação

def is_new_detection(cx, cy):
    for (mcx, mcy), frames in list(memory.items()):
        dist = np.sqrt((mcx - cx) ** 2 + (mcy - cy) ** 2)
        if dist < min_dist:
            return False
    return True

def update_memory():
    for key in list(memory.keys()):
        memory[key] -= 1
        if memory[key] <= 0:
            del memory[key]

def validate_detection(cx, cy):
    buffer.append((cx, cy))
    if len(buffer) > buffer_size:
        buffer.pop(0)
    occurrences = sum(1 for (bx, by) in buffer if np.sqrt((bx - cx) ** 2 + (by - cy) ** 2) < min_dist)
    return occurrences >= (buffer_size - 2)

def check_tombamento(contours):
    global latas_tombadas
    tombadas = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        # Consideramos que latas em pé têm aspecto próximo de 1
        if aspect_ratio < 0.8 or aspect_ratio > 1.2:  # Aspecto diferente de 1 indica tombamento
            latas_tombadas += 1
            tombadas.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Marca a lata tombada em vermelho
    return tombadas

def set_info(detected_cans):
    global latas
    for (cx, cy) in detected_cans:
        if validate_detection(cx, cy) and is_new_detection(cx, cy) and (linha_ROI - 10) < cy < (linha_ROI + 10):
            latas += 1
            memory[(cx, cy)] = max_memory_frames
            cv2.line(frame, (25, linha_ROI), (1200, linha_ROI), (0, 127, 255), 3)
            print(f"Latas detectadas até o momento: {latas}")

def show_info(frame, mask):
    text = f'Latas: {latas} | Tombadas: {latas_tombadas}'
    cv2.putText(frame, text, (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Vídeo Original", frame)

cap = cv2.VideoCapture(VIDEO)
background_subtractor = Subtractor(algorithm_type)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    mask = background_subtractor.apply(frame)
    mask = Filter(mask, 'combine')

    # Detectar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_cans = []
    tombadas = check_tombamento(contours)  # Verifica latas tombadas

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        # Considera que latas em pé têm aspecto próximo de 1
        if 0.8 <= aspect_ratio <= 1.2:  # Latas em pé
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Marca a lata em pé com verde
            detected_cans.append((x + w // 2, y + h // 2))  # Adiciona o centro da lata detectada

    # Linha de contagem
    cv2.line(frame, (25, linha_ROI), (1200, linha_ROI), (255, 127, 0), 3)

    set_info(detected_cans)
    show_info(frame, mask)

    update_memory()

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
