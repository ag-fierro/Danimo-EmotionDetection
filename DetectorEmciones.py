import cv2
from deepface import DeepFace

# Configuración
cap = cv2.VideoCapture(0)

backends = [
    'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
    'retinaface', 'mediapipe', 'yolov8n', 'yolov8m', 
    'yolov8l', 'yolov11n', 'yolov11s', 'yolov11m',
    'yolov11l', 'yolov12n', 'yolov12s', 'yolov12m',
    'yolov12l', 'yunet', 'centerface',
]
detector = 'mediapipe'

frame_interval = 3  # analizar 1 de cada 5 frames
frame_count = 0
last_emotions = []  # almacena emociones recientes para mostrar


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # espejo
    
    frame_count += 1

    # Analizar solo cada N frames
    if frame_count % frame_interval == 0:
        try:
            results = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'  # más rápido que RetinaFace
            )

            # Guardar última emoción detectada
            if results:
                last_emotions = []
                for res in results:
                    last_emotions.append((res['region'], res['dominant_emotion']))
        except Exception as e:
            print("Error analizando emociones:", e)

    # Dibujar emociones en la pantalla
    for region, emotion in last_emotions:
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar la ventana
    cv2.imshow('DeepFace Emotion Detector', frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
