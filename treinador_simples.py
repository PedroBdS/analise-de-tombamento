import gdown
import PIL
from PIL import Image
import pathlib
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

caminho = "./treino/dataset_latas_tombadas"

data_dir = pathlib.Path(caminho)

totalDeImagens = len(list(data_dir.glob("*/*.jpg")))
subfolders = [f.name for f in data_dir.iterdir() if f.is_dir()]
TotalDeClasses = len(subfolders)

print(f'Total de imagens: {totalDeImagens}')

print(f'Total de classes: {TotalDeClasses}')

print(f'Classes: {", ".join(pasta for pasta in subfolders)}')

leafblight = list(data_dir.glob('empe/*'))

for subfolder in subfolders:
    path = data_dir / subfolder
    images = list(path.glob('*.jpg'))
    print(f"Classe '{subfolder}' tem {len(images)} imagens.")

    # Verificando as dimensões e canais de uma imagem exemplo
    if images:
        img = PIL.Image.open(str(images[0]))
        img_array = np.array(img)
        print(f"Dimensões da primeira imagem em '{subfolder}': {img_array.shape}")

batch_size = 64
altura = 150
largura = 150

'''
treino = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=568,
    image_size=(altura,largura),
    batch_size=batch_size
)

print(treino.class_names)
'''

'''
validacao = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=568,
    image_size=(altura,largura),
    batch_size=batch_size
)
'''
def plota_resultados(history,epocas):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  intervalo_epocas = range(epocas)
  plt.figure(figsize=(12, 6))
  plt.subplot(1, 2, 1)
  plt.plot(intervalo_epocas, acc, label='Acurácia do Treino')
  plt.plot(intervalo_epocas, val_acc, label='Acurácia da Validação')
  plt.legend(loc='lower right')


  plt.subplot(1, 2, 2)
  plt.plot(intervalo_epocas, loss, label='Custo do Treino')
  plt.plot(intervalo_epocas, val_loss, label='Custo da Validação')
  plt.legend(loc='upper right')
  plt.show()

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 0.93):
            print("\n Alcançamos 93% de acurácia. Parando o treinamento!")
            self.model.stop_training = True

callbacks = myCallback()

'''
data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.05),
  ]
)
'''

'''
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(altura, largura, 3)),
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    # Add convolutions and max pooling
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)
])

modelo.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
n_classes

'''

'''
epocas = 10

history = modelo.fit(
    treino,
    validation_data=validacao,
    epochs=epocas,

)
plota_resultados(history,epocas)
'''

'''
ultima_camada = modelo.get_layer('mixed7')
print('tamanho da última camada: ', ultima_camada.output_shape)
ultima_saida = ultima_camada.output
'''

'''
modelo.save('/content/drive/My Drive/indentificacao_de_latas1.h5')
'''