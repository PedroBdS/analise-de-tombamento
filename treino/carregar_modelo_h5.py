import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import keras

path_model = "./treino/modelos/alura_tombamento_modelo.h5"

path_img = "C:/Users/PBOTTARI/Documents/Python/tombamento/analise-se-tombamento/tomada3.jpeg"

def carrega_modelo(path_model):
    
    interpreter = keras.models.load_model(path_model)

    return interpreter

def carrega_imagem(path_img):
    
    # Cria um file uploader que permite o usuário carregar imagens
    
    if path_img is not None:
        
        # image_data = Image.open(path_img)
        image = Image.open(path_img)
        # image = Image.open(io.BytesIO(image_data))

        #Pré-processamento da imagem
        image = np.array(image, dtype=np.float32)
        image = image / 255.0  # Normalização para o intervalo [0, 1]
        image = np.expand_dims(image, axis=0)

        return image

def previsao(interpreter,image):
# Realiza a previsão diretamente com o modelo Keras
    output_data = interpreter.predict(image)

    # Classes de saída (ajustar conforme necessário)
    classes = ['tombada', 'empe', "outraclasse"]

    # Formata os resultados em um DataFrame
    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100 * output_data[0]
    
    # Cria o gráfico com Plotly
    fig = px.bar(df, y='classes', x='probabilidades (%)', orientation='h', text='probabilidades (%)',
                 title='Probabilidade de haver lata tombada')
    
    fig.show()

def main():    

    interpreter = carrega_modelo(path_model)

    image = carrega_imagem(path_img)

    if image is not None:

        previsao(interpreter,image) 
    


if __name__ == "__main__":
    main()

# imagem = Image.open(path_img)

# imagem.show()