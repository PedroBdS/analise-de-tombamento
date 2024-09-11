import cv2
import numpy as np
from keras.models import load_model

# Carregar o modelo
model = load_model('C:/Users/PBOTTARI/Documents/Python/tombamento/analise-se-tombamento/modelo_latas_tombadas.h5')

# Função para pré-processar o frame
def preprocess_frame(frame):
    # Redimensionar o frame para o tamanho esperado pelo modelo
    resized_frame = cv2.resize(frame, (224, 224))  # Ajuste o tamanho conforme seu modelo
    # Normalizar o frame
    normalized_frame = resized_frame / 255.0
    # Adicionar uma dimensão extra para o batch
    return np.expand_dims(normalized_frame, axis=0)

# Função para detectar o objeto
def detect_object(frame):
    processed_frame = preprocess_frame(frame)
    # Realizar a inferência
    prediction = model.predict(processed_frame)
    # Aqui você deve adicionar a lógica para interpretar a previsão
    # Para este exemplo, vamos supor que a previsão é a coordenada central e o raio
    x_center, y_center, radius = prediction[0]  # Ajuste conforme a saída do seu modelo
    return (int(x_center), int(y_center), int(radius))

# Abrir o vídeo
video_capture = cv2.VideoCapture('./teste5.mp4')
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Definir o codec e criar o objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video_anotado.mp4', fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Detectar o objeto
    x_center, y_center, radius = detect_object(frame)
    
    # Desenhar um círculo ao redor do objeto detectado
    cv2.circle(frame, (x_center, y_center), radius, (0, 255, 0), 2)
    
    # Escrever o frame anotado no vídeo de saída
    out.write(frame)

# Liberar recursos
video_capture.release()
out.release()
cv2.destroyAllWindows()
