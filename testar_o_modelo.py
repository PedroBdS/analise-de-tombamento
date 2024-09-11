import cv2
import numpy as np
from keras.models import load_model

# Carregar o modelo salvo
model = load_model('./modelo_latas_tombadas.h5')

# Função para processar cada frame do vídeo
def process_frame(frame):
    # Redimensionar o frame para o tamanho esperado pelo modelo
    input_size = (224, 224)  # Tamanho do input que o modelo espera, ajuste conforme necessário
    resized_frame = cv2.resize(frame, input_size)
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Normalizar o frame
    frame_normalized = frame_rgb / 255.0
    frame_input = np.expand_dims(frame_normalized, axis=0)
    
    # Fazer a predição
    prediction = model.predict(frame_input)
    
    # Verificar a posição do objeto previsto (ajuste conforme a saída do seu modelo)
    x, y, w, h = prediction[0]  # Supondo que a saída seja as coordenadas (x, y, largura, altura)

    # Desenhar um círculo em torno do objeto
    center_x, center_y = int(x + w / 2), int(y + h / 2)
    radius = int(max(w, h) / 2)
    cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)
    
    return frame

# Abrir o vídeo
video_path = './teste5.mp4'
cap = cv2.VideoCapture(video_path)

# Verificar se o vídeo abriu corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Definir o codec e criar o objeto de gravação de vídeo para salvar a saída
output_path = 'video_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Processar o frame e identificar o objeto
    frame_with_detections = process_frame(frame)
    
    # Mostrar o frame com as detecções (opcional)
    cv2.imshow('Detecção de Objetos', frame_with_detections)
    
    # Gravar o frame no vídeo de saída
    out.write(frame_with_detections)
    
    # Sair ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
out.release()
cv2.destroyAllWindows()
   
