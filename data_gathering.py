import cv2
import mediapipe as mp
import os
import numpy as np

#----------------------------- Creamos la carpeta donde almacenaremos el entrenamiento ---------------------------------
#nombre = 'hora'
#direccion = r'C:\Users\JuliaElenaSilloCondo\OneDrive - ITS Angelo Rizzoli\Documenti\Deep_learning\Julia_Sillo_esameDL\data\validation'
#carpeta = direccion + '/' + nombre
#if not os.path.exists(carpeta):
#    print('Carpeta creada: ',carpeta)
#    os.makedirs(carpeta)

#Contador para el nombre de la fotos
cont = 0
#leer la camara
cap = cv2.VideoCapture(0)

#----------------------------Creamos un objeto que va almacenar la deteccion y el seguimiento de las manos------------
clase_manos  =  mp.solutions.hands
manos = clase_manos.Hands() #Primer parametro, FALSE para que no haga la deteccion 24/7
                            #Solo hara deteccion cuando hay una confianza alta
                            #Segundo parametro: numero maximo de manos
                            #Tercer parametro: confianza minima de deteccion
                            #Cuarto parametro: confianza minima de seguimiento

#----------------------------------Metodo para dibujar las manos---------------------------
dibujo = mp.solutions.drawing_utils #Con este metodo dibujamos 21 puntos criticos de la mano
posiciones=[]
arrays=[]
while True:
    ret,frame = cap.read()# comentar
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #comentar
    copia = frame.copy() #comentar
    resultado = manos.process(color) #comentar
    posiciones = []  # En esta lista vamos a almacenar las coordenadas de los puntos
                    #print(resultado.multi_hand_landmarks) #Si queremos ver si existe la deteccion

    if resultado.multi_hand_landmarks: #Si hay algo en los resultados entramos al if
        for mano in resultado.multi_hand_landmarks:  #Buscamos la mano dentro de la lista de manos que nos da el descriptor
            for id, lm in enumerate(mano.landmark):  #Vamos a obtener la informacion de cada mano encontrada por el ID
                print(id,lm) #Como nos entregan decimales (Proporcion de la imagen) debemos pasarlo a pixeles
                alto, ancho, c = frame.shape  #Extraemos el ancho y el alto de los fotogramas para multiplicarlos por la proporcion
                corx, cory = int(lm.x*ancho), int(lm.y*alto) #Extraemos la ubicacion de cada punto que pertence a la mano en coordenadas
                posiciones.append([id,corx,cory])
            dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)
            if len(posiciones) != 0:
                xmin = min(posiciones, key=lambda x: x[1])[1]
                ymin = min(posiciones, key=lambda x: x[2])[2]
                xmax = max(posiciones, key=lambda x: x[1])[1]
                ymax = max(posiciones, key=lambda x: x[2])[2]
                dedos_reg = copia[ymin:ymax, xmin:xmax]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            dedos_reg = cv2.resize(dedos_reg,(200,200), interpolation = cv2.INTER_CUBIC) #Redimensionamos las fotos
            arrays.append(dedos_reg)
            #cv2.imwrite(carpeta + "/Mano_{}.jpg".format(cont),dedos_reg)
            cont = cont + 1
            print(dedos_reg)
    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27 or len(arrays) >= 300:
        break
    
cap.release()
cv2.destroyAllWindows()

arrays_np = np.array(arrays)
print(arrays_np.shape)

import matplotlib.pyplot as plt

# Supongamos que 'array' es el array que contiene las imágenes
imagen = arrays_np[0]  # Obtén la primera imagen del array

# Muestra la imagen utilizando Matplotlib
plt.imshow(imagen)
plt.axis('off')  # Desactiva los ejes
plt.show()
