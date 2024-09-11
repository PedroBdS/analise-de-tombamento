import tensorflow as tf
# from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
from keras import layers, models
import matplotlib.pyplot as plt
import os

# Definir caminho para o dataset de latas tombadas
train_dir = 'C:/Users/PBOTTARI/Downloads/dataset_latas_tombadas'  # Substitua pelo caminho correto

# Definir parâmetros
img_width, img_height = 150, 150  # Tamanho padrão para imagens
batch_size = 32
epochs = 10

# Criar geradores de dados para carregar e aumentar os dados
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalização dos pixels para [0, 1]
    shear_range=0.2,             # Aumento: Aplicação de shear
    zoom_range=0.2,              # Aumento: Aplicação de zoom
    horizontal_flip=True,        # Aumento: Inversão horizontal
    validation_split=0.2         # Divisão de 20% dos dados para validação
)

# Gerador de dados de treino - apenas latas tombadas
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',          # Uma única classe: latas tombadas
    subset='training'             # Conjunto de treino
)

# Gerador de dados de validação - apenas latas tombadas
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'           # Conjunto de validação
)

# Definindo a arquitetura da rede CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Sigmoid para saída binária
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Exibir o resumo do modelo
model.summary()

# Treinar o modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Salvar o modelo treinado para uso futuro
model.save('modelo_classificacao_latas_tombadas.h5')

# Plotar a precisão do modelo durante o treinamento
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Acurácia de Treinamento')
plt.plot(epochs_range, val_acc, label='Acurácia de Validação')
plt.legend(loc='lower right')
plt.title('Acurácia de Treinamento e Validação')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perda de Treinamento')
plt.plot(epochs_range, val_loss, label='Perda de Validação')
plt.legend(loc='upper right')
plt.title('Perda de Treinamento e Validação')
plt.show()
