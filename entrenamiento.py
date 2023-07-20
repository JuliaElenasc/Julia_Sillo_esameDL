import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import imgaug.augmenters as iaa
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import seaborn as sns
import mediapipe as mp
from keras.preprocessing.image import img_to_array



def preprocess_coordinates(coordinates):
    # Obtener el valor mínimo y máximo para cada coordenada (eje x e y por separado)
    min_vals = np.min(coordinates, axis=0)
    max_vals = np.max(coordinates, axis=0)

    # Escalar las coordenadas al rango [0, 1]
    normalized_coordinates = (coordinates - min_vals) / (max_vals - min_vals)

    return normalized_coordinates




def normalize_data(data_dict):
    normalized_data_dict = {}
    for label, data in data_dict.items():
        # Obtener el valor mínimo y máximo para cada coordenada
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        # Escalar los datos al rango [0, 1]
        normalized_data = (data - min_val) / (max_val - min_val)
        normalized_data_dict[label] = normalized_data
    return normalized_data_dict


with open('tensor_dict.pkl', 'rb') as file:
    tensor_dict = pickle.load(file)

print(tensor_dict)


datos_lista = []
for etiqueta, coordenadas in tensor_dict.items():
    for coordenada in coordenadas:
        datos_lista.append((coordenada, etiqueta))

coordenadas, etiquetas = zip(*datos_lista)

# Convertir las listas de coordenadas y etiquetas a arreglos numpy
coordenadas = np.array(coordenadas)
etiquetas = np.array(etiquetas)

#NORMALIZAR LOS DATOS

normalized_data_dict = normalize_data(tensor_dict)
print(normalized_data_dict)

for etiqueta, array in normalized_data_dict.items():
    print("Etiqueta:", etiqueta)
    print("Tipo de dato del array:", array.dtype)
    print("============================================")

X_train = []
X_test = []
X_validation = []

y_train = []
y_test = []
y_validation = []

for etiqueta, array in normalized_data_dict.items():
# Dividir los datos en conjuntos de entrenamiento, prueba y validación (80%, 10%, 10%)
    X_train_etiqueta, X_temp, y_train_etiqueta, y_temp = train_test_split(array, [etiqueta]*len(array), test_size=0.2)
    X_validation_etiqueta, X_test_etiqueta, y_validation_etiqueta, y_test_etiqueta = train_test_split(X_temp, y_temp, test_size=0.5)
    
    # Agregar los datos de la etiqueta actual a las listas generales
    X_train.extend(X_train_etiqueta)
    X_test.extend(X_test_etiqueta)
    X_validation.extend(X_validation_etiqueta)
    
    y_train.extend(y_train_etiqueta)
    y_test.extend(y_test_etiqueta)
    y_validation.extend(y_validation_etiqueta)

# Convertir las listas a arreglos numpy
X_train = np.array(X_train)
X_test = np.array(X_test)
X_validation = np.array(X_validation)

y_train = np.array(y_train)
y_test = np.array(y_test)
y_validation = np.array(y_validation)
y_train = y_train.astype(int)
y_validation = y_validation.astype(int)
y_test = y_test.astype(int)


# Imprimir los tamaños de los conjuntos de datos resultantes
print("Datos de entrenamiento:", len(X_train))
print("Datos de validación:", len(X_validation))
print("Datos de prueba:", len(X_test))

# Diseño de la arquitectura de la red neuronal

modelo = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # Capa Dropout con tasa de 0.2
    tf.keras.layers.Dense(3, activation='softmax')  # 3 neuronas en la capa de salida para 3 clases
])


# Compilar el modelo y entrenarlo con los datos aumentados
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')
])
# Compilar el modelo
modelo.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



# Entrenar el modelo
history = modelo.fit(X_train, y_train,
                       validation_data=(X_validation, y_validation),
                       batch_size=32,
                       epochs=60)

# Evaluar el modelo
score = modelo.evaluate(X_test, y_test)

# Crear gráficos de los datos de entrenamiento y validación
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


#guardar el modelo
#modelo.save_weights('pesos.h5')
#modelo.save('modelo1.h5')

y_test = to_categorical(y_test, num_classes=3)
y_pred = modelo.predict(X_test)

y_test_classes = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)
confusion_mtx = confusion_matrix(y_test_classes, y_pred_classes)

# Crear un heatmap de la matriz de confusión
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicha')
plt.ylabel('Verdadera')
plt.title('Confusion Matrix')
plt.show()

'''# Definir las etiquetas de clase

class_labels = {0: 'saludo', 1: 'hora', 2: 'ok'}

# Capturar video en tiempo real
cap = cv2.VideoCapture(0)
clase_manos = mp.solutions.hands
manos = clase_manos.Hands()
dibujo = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break

    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            for id, lm in enumerate(mano.landmark):
                alto, ancho, c = frame.shape
                corx, cory = int(lm.x * ancho), int(lm.y * alto)
                posiciones.append([id, corx, cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)

        if len(posiciones) >= 18:  # Comprobamos que haya suficientes puntos detectados
            pto_i5 = posiciones[9]
            x1, y1 = (pto_i5[1] - 80), (pto_i5[2] - 80)
            dedos_reg = copia[y1:y1 + 160, x1:x1 + 160]  # Ajustar la región de interés
            dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)
            x = img_to_array(dedos_reg)
            x = np.expand_dims(x, axis=0)

            # Preprocesar la imagen antes de pasarla al modelo
            x = x / 255.0  # Normalizar los valores de píxeles entre 0 y 1

            # Aquí realizamos la predicción usando el modelo
            vector = modelo.predict(x)
            respuesta = np.argmax(vector)

            if respuesta == 1:
                color_rect = (0, 255, 0)  # Verde
                texto = class_labels[1]
            else:
                color_rect = (0, 0, 255)  # Rojo
                texto = class_labels[0]

            cv2.rectangle(frame, (x1, y1), (x1 + 160, y1 + 160), color_rect, 3)
            cv2.putText(frame, texto, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color_rect, 1, cv2.LINE_AA)

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()'''




'''# Capturar video en tiempo real

cap = cv2.VideoCapture(0)
while True:
    # Leer el frame actual del video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Procesar el frame para obtener las coordenadas de los dedos
    processed_frame = preprocess_coordinates(frame)

    # Realizar la predicción con el modelo
    prediction = modelo.predict(frame)

    # Obtener la mano correspondiente a la predicción
    mano = class_labels[np.argmax(prediction)]

    # Agregar una etiqueta al video que muestre la mano correspondiente
    cv2.putText(frame, mano, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el frame procesado en la pantalla
    cv2.imshow('Video', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar la ventana
cap.release()
cv2.destroyAllWindows()'''