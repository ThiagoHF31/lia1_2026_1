import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Download do modelo do MediaPipe para a API de Tasks
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Baixando o modelo 'hand_landmarker.task'...")
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    urllib.request.urlretrieve(url, model_path)
    print("Download concluído!")

# Inicialização da captura de vídeo
video = cv2.VideoCapture(0)

# Configuração do MediaPipe Hands (Tasks API para Python 3.12+)
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Conversão da imagem de BGR (câmera) para RGB e criação do mp.Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # Processamento com a nova API
    detection_result = detector.detect(mp_image)
    hand_landmarks_list = detection_result.hand_landmarks
    
    height, width, _ = frame.shape

    if hand_landmarks_list:
        for landmarks in hand_landmarks_list:
            
            # Extração das coordenadas dos pontos
            points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

            # Desenho manual das conexões e dos pontos (substituindo mp.solutions)
            for connection in HAND_CONNECTIONS:
                cv2.line(frame, points[connection[0]], points[connection[1]], (0, 255, 0), 2)
            for point in points:
                cv2.circle(frame, point, 4, (0, 0, 255), -1)

            # Contagem dos dedos levantados
            finger_tips = [8, 12, 16, 20]
            count = 0

            if points[4][0] < points[2][0]:
                count += 1
            count += sum(1 for tip in finger_tips if points[tip][1] < points[tip - 2][1])

            # Desenho do retângulo e contagem na imagem
            cv2.rectangle(frame, (80, 10), (200, 110), (255, 0, 0), -1)
            cv2.putText(frame, str(count), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)

    cv2.imshow('Imagem', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()