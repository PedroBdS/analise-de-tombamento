import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam


# Configurações
IMG_SIZE = 224  # Tamanho da imagem para a MobileNetV2
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = './dataset_latas_tombadas'  # Caminho para as imagens de treinamento

# Preparar o data generator para as imagens de treinamento
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'  # 'binary' para dois classes ou 'categorical' para mais de duas
)

# Carregar o modelo MobileNetV2 pré-treinado sem a camada de classificação final
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Adicionar camadas personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Construir o modelo
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar as camadas base
for layer in base_model.layers:
    layer.trainable = False

# Compilar o modelo
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(train_generator, epochs=EPOCHS)

# Salvar o modelo treinado
model.save('object_recognition_model.h5')
