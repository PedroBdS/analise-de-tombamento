import numpy as np
import cv2
import sys

# Definindo o vídeo e algoritmo de subtração de fundoq
VIDEO = "./teste5.mp4"
algorithm_type = 'MOG2'  # Usando apenas um algoritmo de subtração

# Função para escolher o subtrator de fundo
def Subtractor(algorithm_type):
    if algorithm_type == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if algorithm_type == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if algorithm_type == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    if algorithm_type == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if algorithm_type == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    print('Erro - Insira uma nova informação')
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO)

# Selecionando o algoritmo de subtração
background_subtractor = Subtractor(algorithm_type)

# Valores de raio mínimo e máximo para detecção de círculos
min_radius = 10
max_radius = 40

def main():
    while cap.isOpened():
        ok, frame = cap.read()

        if not ok:
            print('Frames acabaram!')
            break

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Aplicando o subtrator de fundo
        fg_mask = background_subtractor.apply(frame)

        # Melhorando a imagem da máscara
        blurred = cv2.GaussianBlur(fg_mask, (1, 1), 0)
        _, thresh = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY)

        # Encontrando contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # Aproximando a forma para verificar se é circular
            area = cv2.contourArea(cnt)
            if area > 10:  # Ignorar ruídos menores
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                if min_radius <= radius <= max_radius:
                    # Desenhar círculo na imagem original
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

        # Exibir frames
        cv2.imshow('Original', frame)
        cv2.imshow('Máscara', fg_mask)

        # Sair com a tecla "c"
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

main()
