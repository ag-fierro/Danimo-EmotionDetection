import os
import random

import time
from time import sleep

import cv2
from deepface import DeepFace
import numpy as np

import tkinter as tk
from PIL import Image, ImageTk

IMG_PATH = "src/img/"

FD_THRESHOLD = 0.7  # umbral de confianza para detección facial
DS_THRESHOLD = 0.7  # probabilidad de que "DANI diga"

FRAME_INTERVAL = 10  # analizar 1 de cada X frames

ANCHO_VENTANA = 1600
ALTO_VENTANA = 900

PROPORCION_VIDEO = 0.50

SLEEP_TIME = 0.3  # tiempo de espera tras cada ronda

EMOTIONS = ['Alegria', 'Tristeza', 'Enojo', 'Sorpresa', 'Miedo']

EMOTIONS_TRANSLATED = {
    'happy': 'Alegria',
    'sad': 'Tristeza',
    'angry': 'Enojo',
    'surprise': 'Sorpresa',
    'fear': 'Miedo',
    'disgust': 'Discgusto',
    'neutral': 'Neutral'
}

EMOTION_SPRITES = {
    "happy": "alegria.png",
    "sad": "tristeza.png",
    "angry": "enojo.png",
    "surprise": "sorpresa.png",
    "neutral": "neutral.png",
    "fear": "miedo.png",
    "disgust": "disgusto.png"
}

WEIGHTS = {
    'angry': 1.0,
    'disgust': 1.8,
    'fear': 1.0,
    'happy': 1.0,
    'sad': 1.0,
    'surprise': 1.5,
    'neutral': 1.3
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
    
    global frame_count,last_emotions,start_time,current_emotion,level,dani_dice,detected,level
    
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return
          
    frame_count += 1
      
    frame = cv2.flip(frame, 1)  # espejo
    frame = cv2.resize(frame, ( label.winfo_width(),label.winfo_height() ) )
    
    #### CODIGO INTERNO
    
    # Mostrar temporizador
    time_left = int(TIME_TO_RESPOND - (time.time() - start_time))
    
        # Analizar solo cada N frames
    if frame_count % FRAME_INTERVAL == 0:
        last_emotions = analyze(WEIGHTS, frame)
        
        if last_emotions and len(last_emotions) > 0:
            print(f"Emociones detectadas: {[e[1] for e in last_emotions]}")    
            detected = EMOTIONS_TRANSLATED[last_emotions[0][1]] 
        
    orden = ""
    # Mostrar gesto actual e interpretado
    if (dani_dice):
        orden += "Dani dice: " 
    
    orden += f"{current_emotion.upper()}"
    
    label_text.config(text=f"Tiempo restante: {time_left}s  Nivel: {level} \n {orden.upper()} \n Actual: {detected.upper()} ")

    # Dibujar emociones en la pantalla
    for region, emotion in last_emotions:
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        if emotion in loaded_emojis:
            emoji = loaded_emojis[emotion]
            frame = overlay_image(frame, emoji, x, y, scale=w/emoji.shape[1])
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
    # Comprobar si debo cambiar emoción            
    change_emotion = time.time() - start_time > TIME_TO_RESPOND or detected == current_emotion
    
    if (dani_dice):
        if time.time() - start_time > TIME_TO_RESPOND:
            print(" ")
            #add_output(" Tiempo agotado. Reiniciando juego.")
            cv2.putText(frame, f"Tiempo agotado. Reiniciando juego.", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            sleep(SLEEP_TIME)
            level = 1
            
        elif detected == current_emotion:
            print("Correcto!")
            
            cv2.putText(frame, f"Correcto!", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            #add_output("Correcto!")
            sleep(SLEEP_TIME)
            level += 1
    else:
        if time.time() - start_time > TIME_TO_RESPOND:
            print("Correcto!")
            cv2.putText(frame, f"Correcto!", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            sleep(SLEEP_TIME)
            #add_output("Correcto!")
            level += 1
        elif detected == current_emotion:
            print(" Simon No LO DIJO. Reiniciando juego.")
            cv2.putText(frame, f"Simon No LO DIJO. Reiniciando juego.", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            sleep(SLEEP_TIME)
            #add_output(" Simon No LO DIJO. Reiniciando juego.")
            level = 1
            
    if (change_emotion):
        current_emotion, dani_dice = new_emotion(DS_THRESHOLD)

        start_time = time.time() 

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
    
    label_text.configure(x=ancho/3, y=30)

def pantalla_inicio():
    path = os.path.join(IMG_PATH, "fondo.png")
    pantalla = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    pantalla = cv2.resize(pantalla, (ANCHO_VENTANA, ALTO_VENTANA))  
    
    reglas_base = [
        "Reglas del juego:",
        "- Si dice 'DANI dice', intenta mostrar esa emocion.",
        "- Si NO dice 'DANI dice', no lo hagas",
        "- Si fallas, vuelves al nivel 1.",
        "- Si aciertas, subes de nivel.",
        "- La camara debe captar tu rostro",
        "",
        "",
        "Selecciona la dificultad y el juego comenzara:",
        "1 - Facil (8s)    2 - Normal (3s)    3 - Dificil (1s)",
        "",
        "Pulsa 'ESC' para salir..."
    ]

    tiempo = 8  # por defecto

    while True:

        y = 120
        
        cv2.putText(pantalla, "Bienvenido a DANI Dice", (int(ANCHO_VENTANA/3)+50, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 3)
        
        y += 50
        
        for regla in reglas_base:
            cv2.putText(pantalla, regla, (int(ANCHO_VENTANA/3)+50, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 2)
            y += 35

        cv2.imshow("DANI Dice", pantalla)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):            
            tiempo = 8
            cv2.destroyAllWindows()
            return tiempo
        elif key == ord('2'):            
            tiempo = 3
            cv2.destroyAllWindows()
            return tiempo
        elif key == ord('3'):            
            tiempo = 1
            cv2.destroyAllWindows()
            return tiempo
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            exit()

def new_emotion(DS_THRESHOLD):
    current_emotion = random.choice(EMOTIONS)
    prob_dani_dice = random.random()
    dani_dice = False
    if(DS_THRESHOLD > prob_dani_dice):
        dani_dice = True
    
    return current_emotion,dani_dice

TIME_TO_RESPOND = pantalla_inicio()
     
print(f"Tiempo para responder: {TIME_TO_RESPOND} segundos")


# Configuración camara
cap = cv2.VideoCapture(0)

frame_count = 0 

last_emotions = []  # almacena emociones detectadas la ultima vez


loaded_emojis = {}
for emotion, file in EMOTION_SPRITES.items():
    path = os.path.join(IMG_PATH, file)
    if os.path.exists(path):
        loaded_emojis[emotion] = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # con canal alpha
        
current_emotion, dani_dice = new_emotion(DS_THRESHOLD)
detected = 'none'
level = 1
start_time = time.time()


### VENTANA TKINTER ###

root = tk.Tk()
root.title("DANI DICE - DETECCIÓN DE EMOCIONES")
root.geometry(f"{ANCHO_VENTANA}x{ALTO_VENTANA}")

# === Cargar imagen de fondo ===
ruta_fondo = os.path.join(IMG_PATH, "fondo.png")  # Cambiá por tu archivo
imagen_original = Image.open(ruta_fondo)
fondo_tk = ImageTk.PhotoImage(imagen_original)

# === Label que contendrá el fondo ===
label_fondo = tk.Label(root)
label_fondo.place(x=0, y=0, relwidth=1, relheight=1)


texto = f"Tiempo restante: {TIME_TO_RESPOND}s  Nivel: {level} \n "" \n Emocion detectada: {current_emotion} "
label_text = tk.Label(
    label_fondo, 
    text=texto, 
    font=("Arial", 16), 
    fg="black",       # color de la letra
    justify="center"  # centrar texto si tiene varias líneas
)
label_text.place(relx=0.5, rely=0.08, anchor="n")  # parte superior centrado



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

