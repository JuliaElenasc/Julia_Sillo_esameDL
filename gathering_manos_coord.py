import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle


def plot_hand_tracking(frame, finger_coordinates):
    x_values = []
    y_values = []
    z_values = []

    for finger_coord in finger_coordinates:
        # Algunas tuplas pueden tener solo 3 valores, así que manejamos este caso
        if len(finger_coord) == 4:
            finger_name, finger_x, finger_y, finger_z = finger_coord
            x_values.append(finger_x)
            y_values.append(finger_y)
            z_values.append(finger_z)
        elif len(finger_coord) == 3:
            finger_name, finger_x, finger_y = finger_coord
            x_values.append(finger_x)
            y_values.append(finger_y)
            z_values.append(0)  # Agregamos un valor de 0 para z
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, label='X', marker='o')
    plt.plot(y_values, label='Y', marker='o')
    plt.plot(z_values, label='Z', marker='o')
    plt.xlabel('Frames')
    plt.ylabel('Coordenadas')
    plt.title('Coordenadas de los Dedos')
    plt.legend()
    
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        annotated_image = frame.copy()
        image_height, image_width, _ = annotated_image.shape
        results = hands.process(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

def create_hand_landmarks_tensor(time_limit=20):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    cap = cv2.VideoCapture(0)

    start_time = None
    finger_coordinates = []

    while True:
        ret, frame = cap.read()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if start_time is None:
                    start_time = time.time()

                for finger in mp_hands.HandLandmark:
                    finger_name = finger.name  # Agrega el nombre del dedo a la tupla
                    finger_x = hand_landmarks.landmark[finger].x
                    finger_y = hand_landmarks.landmark[finger].y
                    finger_z = hand_landmarks.landmark[finger].z
                    finger_coordinates.append((finger_name, finger_x, finger_y, finger_z))

                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Encuadrar la mano con un rectángulo
                x_min = int(min(hand_landmarks.landmark, key=lambda landmark: landmark.x).x * frame.shape[1])
                y_min = int(min(hand_landmarks.landmark, key=lambda landmark: landmark.y).y * frame.shape[0])
                x_max = int(max(hand_landmarks.landmark, key=lambda landmark: landmark.x).x * frame.shape[1])
                y_max = int(max(hand_landmarks.landmark, key=lambda landmark: landmark.y).y * frame.shape[0])
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or (start_time is not None and time.time() - start_time >= time_limit):
            break
    #plot_hand_tracking(frame, finger_coordinates)
    
    tensor_hand_landmarks = np.array([coords[1:] for coords in finger_coordinates])

    cap.release()
    cv2.destroyAllWindows()

    return tensor_hand_landmarks

def plot_finger_coordinates(finger_coordinates):
    x_values = []
    y_values = []
    z_values = []

    for finger_coord in finger_coordinates:
        # Algunas tuplas pueden tener solo 3 valores, así que manejamos este caso
        if len(finger_coord) == 4:
            finger_name, finger_x, finger_y, finger_z = finger_coord
            x_values.append(finger_x)
            y_values.append(finger_y)
            z_values.append(finger_z)
        elif len(finger_coord) == 3:
            finger_name, finger_x, finger_y = finger_coord
            x_values.append(finger_x)
            y_values.append(finger_y)
            z_values.append(0)  # Agregamos un valor de 0 para z

    # Visualización de las coordenadas x, y, z de los dedos
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, label='X', marker='o')
    plt.plot(y_values, label='Y', marker='o')
    plt.plot(z_values, label='Z', marker='o')
    plt.xlabel('Frames')
    plt.ylabel('Coordenadas')
    plt.title('Coordenadas de los Dedos')
    plt.legend()
    plt.show()

saludo_tensor = create_hand_landmarks_tensor()
plot_finger_coordinates(saludo_tensor)

hora_tensor = create_hand_landmarks_tensor()
plot_finger_coordinates(hora_tensor)

ok_tensor = create_hand_landmarks_tensor()
plot_finger_coordinates(ok_tensor)
 
#que_tensor = create_hand_landmarks_tensor()


tensor_dict = {
    '0': saludo_tensor,#saludo
    '1': hora_tensor, # hora
    '2': ok_tensor, #ok
    #'3': que_tensor#que
    }

with open('tensor_dict.pkl', 'wb') as f:
    pickle.dump(tensor_dict, f)
print("Diccionario guardado en el archivo 'tensor_dict.pkl'.")
          


    
    
    
    

