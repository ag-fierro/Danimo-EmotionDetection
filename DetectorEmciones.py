import os
import cv2
from deepface import DeepFace

def overlay_image(bg, fg, x, y, scale=1.0):
    """
    Superpone la imagen 'fg' (con transparencia) sobre 'bg' en la posición (x, y).
    
    bg    : imagen de fondo (frame de OpenCV)
    fg    : imagen a superponer (debe tener canal alpha)
    x, y  : coordenadas superiores izquierdas donde se pondrá fg
    scale : factor de escala para ajustar tamaño
    """
    # Redimensionar el emoji
    fg = cv2.resize(fg, (0,0), fx=scale, fy=scale)
    h, w, _ = fg.shape
    rows, cols, _ = bg.shape

    # Ajustar si está cerca del borde
    if y+h > rows: h = rows - y
    if x+w > cols: w = cols - x

    # Canal alpha (transparencia)
    alpha = fg[:, :, 3] / 255.0  # 0-1

    # Mezclar cada canal de color
    for c in range(0, 3):
        bg[y:y+h, x:x+w, c] = alpha[:h, :w]*fg[:h, :w, c] + (1-alpha[:h, :w])*bg[y:y+h, x:x+w, c]

    return bg


# Configuración
cap = cv2.VideoCapture(0)

frame_interval = 5  # analizar 1 de cada 5 frames
frame_count = 0
last_emotions = []  # almacena emociones recientes para mostrar

img_path = "src/img/"

emotion_sprites = {
    "happy": "alegria.png",
    "sad": "tristeza.png",
    "angry": "enojo.png",
    "surprise": "miedo.png",
    "neutral": "ansiedad.png",
    "fear": "miedo.png",
    "disgust": "ansiedad.png"
}

weights = {
    'angry': 1.0,
    'disgust': 1.0,
    'fear': 1.0,
    'happy': 1.0,
    'sad': 1.0,
    'surprise': 1.0,
    'neutral': 1.0  # potenciamos neutral
}

loaded_emojis = {}
for emotion, file in emotion_sprites.items():
    path = os.path.join(img_path, file)
    if os.path.exists(path):
        loaded_emojis[emotion] = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # con canal alpha

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
            if results and results[0]["face_confidence"] > 0.8:            
                last_emotions = []
                for res in results:
                    
                    ## boost neutral emotion.
                    
                    print(res["emotion"])
                    
                    print(res["dominant_emotion"])

                    weighted_scores = {k: res["emotion"][k] * weights.get(k, 1.0) for k in res["emotion"]}
                    dominant_emotion = max(weighted_scores, key=weighted_scores.get)

                    print(weighted_scores)

                    print(dominant_emotion)
                    
                    print("-----")
                
                    # Guardar emocion final
                    
                    last_emotions.append((res['region'], dominant_emotion))
            else:
                last_emotions = []
                    
        except Exception as e:
            print("Error analizando emociones:", e)

    # Dibujar emociones en la pantalla
    for region, emotion in last_emotions:
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        if emotion in loaded_emojis:
            emoji = loaded_emojis[emotion]
            frame = overlay_image(frame, emoji, x, y, scale=w/emoji.shape[1])
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Mostrar la ventana
    cv2.imshow('DeepFace Emotion Detector', frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
