import numpy as np
import cv2
import sys

VIDEO = './videos/Esteira_1.mp4'

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
# PARAMETROS
linha_ROI = 200  # Posição da linha de contagem

min_dist = 35  # Distância mínima entre detecções consecutivas da mesma lata

raio_min = 37
raio_max = 43

# raio_medio = 38
# erro_permitido = 2

# Variáveis para rastreamento
detected_cans = []
memory = {}  # Memória para objetos rastreados (evita contagem dupla)
max_memory_frames = 10  # Máximo de frames para lembrar uma lata detectada

buffer = []  # Buffer para validar detecções ao longo de frames
buffer_size = 1  # Tamanho do buffer de frames consecutivos para validação

latas = 0
latas_tombadas = 0  # Contador de latas tombadas

def salvar_matriz_em_arquivo(matriz, nome_arquivo):
    with open(nome_arquivo, 'w') as arquivo:
        lista = []
        for linha in matriz:
            # Seleciona apenas os dois primeiros valores (x, y)
            linha_xy = linha[:2]
            linha_str = ', '.join(map(str, linha_xy))
            linha_str = eval(linha_str)
            lista.append(linha_str)
        arquivo.write(str(lista))

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

def set_info(detected_cans):
    global latas
    for (cx, cy) in detected_cans:
        if validate_detection(cx, cy) and is_new_detection(cx, cy) and (linha_ROI - 10) < cy < (linha_ROI + 10):
            latas += 1
            memory[(cx, cy)] = max_memory_frames
            # print(f"Latas detectadas até o momento: {latas}")

def show_info(frame, mask):
    # text = f'Latas: {latas} | Tombadas: {latas_tombadas}'
    text = f'Latas: {latas}'
    cv2.putText(frame, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 7)
    cv2.imshow("Vídeo Original", frame)

cap = cv2.VideoCapture(VIDEO)
background_subtractor = Subtractor(algorithm_type)

cont = 0
frames_analisados = 3
historico_de_posicoes = []

while cont < frames_analisados:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    mask = background_subtractor.apply(frame)
    mask = Filter(mask, 'combine')

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.47, minDist=70, param1=70, param2=20, minRadius=raio_min, maxRadius=raio_min)
    # circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.3, minDist=70, param1=70, param2=20, minRadius=raio_medio-erro_permitido, maxRadius=raio_medio+erro_permitido)

    detected_cans = []

    if circles is not None:
        circles = np.around(circles[0, :]).astype("int")
        
        for (x, y, r) in circles:

            if (0) < y < (linha_ROI):  # Se passou pela linha de contagem
                cv2.circle(frame, (x, y), r-10, (0, 255, 0), -1)  # Pintar de verde após a contagem

            else:
                cv2.circle(frame, (x, y), r-10, (255, 0, 0), -1)  # Pintar de azul antes da contagem

            detected_cans.append((x, y))
        historico_de_posicoes.append(detected_cans)

        print(len(historico_de_posicoes))
        # print(f'Frame {cont+1}: {historico_de_posicoes}\n')

    cv2.line(frame, (25, linha_ROI), (1200, linha_ROI), (255, 127, 0), 3)

    set_info(detected_cans)
    show_info(frame, mask)

    update_memory()

    if cv2.waitKey(1) == ord('q'):
        break
    
    cont += 1


salvar_matriz_em_arquivo(circles, "posições.txt")

cap.release()
cv2.destroyAllWindows()
