import tensorflow as tf
import pathlib
import PIL
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt

url = 'C:/Users/PBOTTARI/Documents/Python/tombamento/analise-se-tombamento/dataset_latas_tombadas'

data_dir = pathlib.Path(url)

print(len(list(data_dir.glob('*/*.jpg'))))

subfolders = [f.name for f in data_dir.iterdir() if f.is_dir()]
print(subfolders)

latatombada = list(data_dir.glob('tombada/*'))

PIL.Image.open(str(latatombada[1]))

for subfolder in subfolders:
  path = data_dir / subfolder
  images = list(path.glob('*.jpg'))
  print(f'classe {subfolders} tem {len(images)} imagens')

  if images:
    img = PIL.Image.open(str(images[0]))
    img_array = np.array(img)
    print(f'Dimensão da primeira image em {subfolder}: {img_array.shape}')

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.93):
      print("\n Alcançamos 93% de acurácia. Parando o treinamento!")
      self.model.stop_training = True

callbacks = myCallback()

batch_size = 64
altura = 150
largura = 150

treino = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=568,
    image_size=(altura,largura),
    batch_size=batch_size
)

validacao = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=568,
    image_size=(altura,largura),
    batch_size=batch_size
)

print(treino.class_names)

tf.random.set_seed(424242)

data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.05)
  ]
)

input_shape = (150, 150, 3)

modelo_base = tf.keras.applications.InceptionV3(input_shape=input_shape, include_top=False, weights='imagenet')
modelo_base.trainable = False

modelo_base.summary()

rescale = tf.keras.layers.Rescaling((1./255))
treino = treino.map(lambda x, y: (rescale(x),y))
validacao = validacao.map(lambda x, y: (rescale(x),y))

ultima_camada = modelo_base.get_layer('mixed7')
print('ultima_camada',ultima_camada.output_shape)
ultima_saida = ultima_camada.output

x = tf.keras.layers.Flatten()(ultima_saida)

x = tf.keras.layers.Dense(1024, activation='relu')(x)

x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Dense(4, activation='softmax')(x)

modelo = tf.keras.Model(inputs=modelo_base.input,outputs=x)

modelo.summary()

modelo.compile(optimizer = tf.keras.optimizers.Adam(),
                    loss = 'sparse_categorical_crossentropy',
                    metrics=['accuracy'])

epocas = 20

history = modelo.fit(
    treino,
    validation_data=validacao,
    epochs=epocas,
)

modelo_base.save('alura_tombamento_modelo.h5')
# modelo = tf.keras.models.Sequential([
#     tf.keras.layers.Input(shape=(150, 150, 3)),
#     data_augmentation,
#     tf.keras.layers.Rescaling(1./255),
#     # Add convolutions and max pooling
#     tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     tf.keras.layers.Dense(4, activation=tf.nn.softmax)
# ])

# modelo.compile(optimizer = tf.keras.optimizers.Adam(),
#               loss = 'sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# epocas = 50

# history = modelo.fit(
#     treino,
#     validation_data=validacao,
#     epochs=epocas
# )
# print(modelo.summary())

# def plota_resultados(history,epocas):
#     acc = history.history['accuracy']
#     val_acc = history.history['val_accuracy']

#     loss = history.history['loss']
#     val_loss = history.history['val_loss']

#     intervalo_epocas = range(epocas)
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.plot(intervalo_epocas, acc, label='Acurácia do Treino')
#     plt.plot(intervalo_epocas, val_acc, label='Acurácia da Validação')
#     plt.legend(loc='lower right')


#     plt.subplot(1, 2, 2)
#     plt.plot(intervalo_epocas, loss, label='Custo do Treino')
#     plt.plot(intervalo_epocas, val_loss, label='Custo da Validação')
#     plt.legend(loc='upper right')
#     plt.show()

# plota_resultados(history,epocas)

