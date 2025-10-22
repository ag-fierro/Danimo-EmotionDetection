import os
import cv2
from deepface import DeepFace

import tkinter as tk
from PIL import Image, ImageTk

IMG_PATH = "src/img/"

FD_THRESHOLD = 0.6  # umbral de confianza para detección facial

FRAME_INTERVAL = 5  # analizar 1 de cada 5 frames

ANCHO_VENTANA = 800
ALTO_VENTANA = 600

PROPORCION_VIDEO = 0.50

EMOTION_SPRITES = {
    "happy": "alegria.png",
    "sad": "tristeza.png",
    "angry": "enojo.png",
    "surprise": "sorpresa.png",
    "neutral": "neutral.png",
    "fear": "miedo.png",
    "disgust": "ansiedad.png"
}

WEIGHTS = {
    'angry': 1.0,
    'disgust': 1.0,
    'fear': 1.0,
    'happy': 1.0,
    'sad': 1.0,
    'surprise': 1.0,
    'neutral': 1.0  # potenciamos neutral
}


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

def analyze(weights, frame):
    try:
        results = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'  # más rápido que RetinaFace
            )

            # Guardar última emoción detectada
        
        if results and results[0]["face_confidence"] > FD_THRESHOLD:
            last_emotions = []
            for res in results:
                    # Recalcular para evaluar ponderaciones                    
                weighted_scores = {k: res["emotion"][k] * weights.get(k, 1.0) for k in res["emotion"]}
                dominant_emotion = max(weighted_scores, key=weighted_scores.get)
                
                    # Guardar emocion final
                    
                last_emotions.append((res['region'], dominant_emotion))
        else:
            last_emotions = []
            
        return last_emotions
                    
    except Exception as e:
        print("Error analizando emociones:", e)
    return last_emotions


def update_frame():
    
    global frame_count,last_emotions
    
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return
          
    frame_count += 1
      
    frame = cv2.flip(frame, 1)  # espejo
    frame = cv2.resize(frame, ( label.winfo_width(),label.winfo_height() ) )
    
    #### CODIGO INTERNO

    # Analizar solo cada N frames
    if frame_count % FRAME_INTERVAL == 0:
        last_emotions = analyze(WEIGHTS, frame)

    # Dibujar emociones en la pantalla
    for region, emotion in last_emotions:
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        if emotion in loaded_emojis:
            emoji = loaded_emojis[emotion]
            frame = overlay_image(frame, emoji, x, y, scale=w/emoji.shape[1])
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Mostrar en Tkinter
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    
    # Llamar de nuevo después de 10ms
    root.after(10, update_frame)

# === Función para redimensionar fondo al cambiar tamaño ===
def actualizar_fondo(event):
    global ancho_video, alto_video
    
    ancho = root.winfo_width()
    alto = root.winfo_height()
    
    print(f"Redimensionando fondo a {ancho}x{alto}")  

    imagen_redimensionada = imagen_original.resize((ancho, alto), Image.Resampling.LANCZOS)
    fondo_nuevo = ImageTk.PhotoImage(imagen_redimensionada)
    label_fondo.config(image=fondo_nuevo)
    label_fondo.image = fondo_nuevo  # mantener referencia
    
    label.configure(width=ancho*PROPORCION_VIDEO, height=alto*PROPORCION_VIDEO)
    print(f"Label mide: {label.winfo_width()}x{label.winfo_height()}")
    
     
# Configuración camara
cap = cv2.VideoCapture(0)


frame_count = 0 

last_emotions = []  # almacena emociones recientes para mostrar


loaded_emojis = {}
for emotion, file in EMOTION_SPRITES.items():
    path = os.path.join(IMG_PATH, file)
    if os.path.exists(path):
        loaded_emojis[emotion] = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # con canal alpha


### VENTANA TKINTER ###

root = tk.Tk()
root.title("DANIMO - DETECCIÓN DE EMOCIONES")
root.geometry(f"{ANCHO_VENTANA}x{ALTO_VENTANA}")

# === Cargar imagen de fondo ===
ruta_fondo = os.path.join(IMG_PATH, "fondo.png")  # Cambiá por tu archivo
imagen_original = Image.open(ruta_fondo)
fondo_tk = ImageTk.PhotoImage(imagen_original)

# === Label que contendrá el fondo ===
label_fondo = tk.Label(root)
label_fondo.place(x=0, y=0, relwidth=1, relheight=1)

# === Frame que contendrá el stream de video (encima del fondo) ===
frame_display = tk.Frame(root, bg="black")
frame_display.pack(expand=True, fill='both')
frame_display.place(relx=0.5, rely=0.43, anchor="center")  # centrado

label = tk.Label(frame_display)
label.pack()


# Vincular evento de redimensionado
root.bind("<Configure>", actualizar_fondo)

update_frame()

root.mainloop()
    
cap.release()

