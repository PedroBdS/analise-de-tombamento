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

converter = tf.lite.TFLiteConverter.from_keras_model(modelo)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

modelo_tflite_quantizado = converter.convert()

with open('modelo_quantizado16bits.tflite', 'wb') as f:
    f.write(modelo_tflite_quantizado)

# modelo_base.save('alura_tombamento_modelo.h5')
