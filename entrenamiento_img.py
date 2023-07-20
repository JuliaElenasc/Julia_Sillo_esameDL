import os
import shutil
import random
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras import layers, Sequential, models
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from keras.models import save_model
import mediapipe as mp
from keras.preprocessing.image import img_to_array






# Directorios de las carpetas con las imágenes de los gestos
# Rutas de las carpetas de gestos
carpeta_inicial = r"C:\Users\JuliaElenaSilloCondo\OneDrive - ITS Angelo Rizzoli\Documenti\Deep_learning\Julia_Sillo_esameDL\data"
carpeta_destino=r"C:\Users\JuliaElenaSilloCondo\OneDrive - ITS Angelo Rizzoli\Documenti\Deep_learning\Julia_Sillo_esameDL\divided"
def dividir_imagenes(carpeta_inicial, carpeta_destino):
    # Crear la carpeta datos_divididos si no existe
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    # Obtener la lista de clases (nombres de las subcarpetas) en la carpeta inicial
    clases = os.listdir(carpeta_inicial)

    # Crear las carpetas train, test y validation
    for split in ['train', 'test', 'validation']:
        carpeta_split = os.path.join(carpeta_destino, split)
        if not os.path.exists(carpeta_split):
            os.makedirs(carpeta_split)
        
        # Crear subcarpetas para cada clase dentro de train, test y validation
        for clase in clases:
            carpeta_clase = os.path.join(carpeta_split, clase)
            if not os.path.exists(carpeta_clase):
                os.makedirs(carpeta_clase)

    # Dividir las imágenes para cada clase en train, test y validation
    for clase in clases:
        carpeta_clase_origen = os.path.join(carpeta_inicial, clase)
        imagenes = os.listdir(carpeta_clase_origen)

        # Dividir las imágenes en train, test y validation
        train, test_valid = train_test_split(imagenes, test_size=0.2, random_state=42)
        test, validation = train_test_split(test_valid, test_size=0.5, random_state=42)

        # Mover las imágenes a las carpetas correspondientes
        for archivo in train:
            shutil.copy(os.path.join(carpeta_clase_origen, archivo), os.path.join(carpeta_destino, "train", clase, archivo))

        for archivo in test:
            shutil.copy(os.path.join(carpeta_clase_origen, archivo), os.path.join(carpeta_destino, "test", clase, archivo))

        for archivo in validation:
            shutil.copy(os.path.join(carpeta_clase_origen, archivo), os.path.join(carpeta_destino, "validation", clase, archivo))


# Función para preprocesar la imagen antes de pasarla al modelo
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def preprocess_image(image):
    image = image.resize((150, 150))
    image_array = img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Función para dibujar un rectángulo alrededor de la mano detectada
def draw_hand_rectangle(image, hand_landmarks):
    image_height, image_width, _ = image.shape
    landmarks = hand_landmarks.landmark

    x_min, x_max, y_min, y_max = image_width, 0, image_height, 0

    for landmark in landmarks:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)



if __name__ == '__main__':
    # ------Dividir imagenes en train test validation
    #carpeta_inicial = r"C:\Users\JuliaElenaSilloCondo\OneDrive - ITS Angelo Rizzoli\Documenti\Deep_learning\Julia_Sillo_esameDL\data"
    #carpeta_destino = r"C:\Users\JuliaElenaSilloCondo\OneDrive - ITS Angelo Rizzoli\Documenti\Deep_learning\Julia_Sillo_esameDL\divided"
    #dividir_imagenes(carpeta_inicial, carpeta_destino)

    train_dir = r"C:\Users\JuliaElenaSilloCondo\OneDrive - ITS Angelo Rizzoli\Documenti\Deep_learning\Julia_Sillo_esameDL\divided\train"
    test_dir = r"C:\Users\JuliaElenaSilloCondo\OneDrive - ITS Angelo Rizzoli\Documenti\Deep_learning\Julia_Sillo_esameDL\divided\test"
    val_dir = r"C:\Users\JuliaElenaSilloCondo\OneDrive - ITS Angelo Rizzoli\Documenti\Deep_learning\Julia_Sillo_esameDL\divided\validation"

    input_shape = (150, 150, 3)
    num_classes = len(os.listdir(train_dir))

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    batch_size = 32

    train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

    val_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

    epochs = 10

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator))

    test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
    print('Precisión en el conjunto de prueba:', test_acc)

    # Crear gráficos de los datos de entrenamiento y validación
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    # Obtener las etiquetas reales y predichas
    y_test_classes = test_generator.classes
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Crear la matriz de confusión
    confusion_mtx = confusion_matrix(y_test_classes, y_pred_classes)

    # Crear un heatmap de la matriz de confusión
    sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicha')
    plt.ylabel('Verdadera')
    plt.title('Confusion Matrix')
    plt.show()

    incorrect_indices = np.where(y_test_classes != y_pred_classes)[0]
    # Mostrar las primeras 5 imágenes clasificadas incorrectamente
    num_images_to_show = 5
    for i in range(min(num_images_to_show, len(incorrect_indices))):
        index = incorrect_indices[ i]
        img_path = test_generator.filepaths[index]
        img = Image.open(img_path)
        plt.figure()
        plt.imshow(img)
        true_label = test_generator.classes[index]
        predicted_label = y_pred_classes[index]
        plt.title(f'True Label: {true_label}, Predicted Label: {predicted_label}')
        plt.axis('off')
        plt.show()

    # Guardar modelo
    # save_model(model, 'modelo')

    #-----------------------------verificar clasificacion-------------------
    '''direccion = r'C:\Users\JuliaElenaSilloCondo\OneDrive - ITS Angelo Rizzoli\Documenti\Deep_learning\Julia_Sillo_esameDL/Validacion'
    dire_img = os.listdir(direccion)
    print("Nombres: ", dire_img)
    
    cap=cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    dibujo= mp.solutions.drawing_utils
    while (1):
        ret,frame = cap.read()
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        copia = frame.copy()
        resultado = hands.process(color)
        posiciones = []  # En esta lista vamos a almcenar las coordenadas de los puntos
        #print(resultado.multi_hand_landmarks) #Si queremos ver si existe la deteccion

        if resultado.multi_hand_landmarks: #Si hay algo en los resultados entramos al if
            for hand in resultado.multi_hand_landmarks:  #Buscamos la mano dentro de la lista de manos que nos da el descriptor
                for id, lm in enumerate(hand.landmark):  #Vamos a obtener la informacion de cada mano encontrada por el ID
                    alto, ancho, c = frame.shape  #Extraemos el ancho y el alto de los fotpgramas para multiplicarlos por la proporcion
                    corx, cory = int(lm.x*ancho), int(lm.y*alto) #Extraemos la ubicacion de cada punto que pertence a la mano en coordenadas
                    posiciones.append([id,corx,cory])
                    dibujo.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                if len(posiciones) != 0:
                    pto_i1 = posiciones[3] #5 Dedos: 4 | 0 Dedos: 3 | 1 Dedo: 2 | 2 Dedos: 3 | 3 Dedos: 4 | 4 Dedos: 8
                    pto_i2 = posiciones[17]#5 Dedos: 20| 0 Dedos: 17| 1 Dedo: 17| 2 Dedos: 20| 3 Dedos: 20| 4 Dedos: 20
                    pto_i3 = posiciones[10]#5 Dedos: 12| 0 Dedos: 10 | 1 Dedo: 20|2 Dedos: 16| 3 Dedos: 12| 4 Dedos: 12
                    pto_i4 = posiciones[0] #5 Dedos: 0 | 0 Dedos: 0 | 1 Dedo: 0 | 2 Dedos: 0 | 3 Dedos: 0 | 4 Dedos: 0
                    pto_i5 = posiciones[9]
                    x1,y1 = (pto_i5[1]-80),(pto_i5[2]-80) #Obtenemos el punto incial y las longitudes
                    ancho, alto = (x1+80),(y1+80)
                    x2,y2 = x1 + ancho, y1 + alto
                    dedos_reg = copia[y1:y2, x1:x2]
                    dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)  # Redimensionamos las fotos
                    x = img_to_array(dedos_reg)  # Convertimos la imagen a una matriz
                    x = np.expand_dims(x, axis=0)  # Agregamos nuevo eje
                    vector = model.predict(x)  # Va a ser un arreglo de 2 dimensiones, donde va a poner 1 en la clase que crea correcta
                    resultado = vector[0]  # [1,0] | [0, 1]
                    respuesta = np.argmax(resultado)  # Nos entrega el indice del valor mas alto 0 | 1
                    if respuesta == 1:
                        print(vector,resultado)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(frame, '{}'.format(dire_img[0]), (x1, y1 - 5), 1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)
                    elif respuesta == 0:
                        print(vector, resultado)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, '{}'.format(dire_img[1]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("Video",frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()'''





