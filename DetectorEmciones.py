import os
import random
import json
import time
from time import sleep
from threading import Thread, Lock
from queue import Queue

import cv2
from deepface import DeepFace
import numpy as np

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from variables import *

# ====== VARIABLES GLOBALES INICIALIZADAS ======
frame_count = 0
last_emotions = []
last_capture_time = 0.0
current_emotion = None
detected = 'none'
level = 1
start_time = time.time()
dani_dice = False
combo = 0
max_combo = 0
score = 0
high_score = 0
is_analyzing = False
analysis_queue = Queue()
emotion_lock = Lock()

# Variables para efectos visuales
flash_color = None
flash_start = 0
show_feedback = False
feedback_text = ""
feedback_color = "black"

def overlay_image(bg, fg, x, y, scale=1.0):
    """
    Superpone la imagen 'fg' (con transparencia) sobre 'bg' en la posición (x, y).
    """
    # Redimensionar el emoji - AUMENTADO 50% para mejor visibilidad
    fg = cv2.resize(fg, (0,0), fx=scale*1.5, fy=scale*1.5)
    h, w, _ = fg.shape
    rows, cols, _ = bg.shape

    # Ajustar si está cerca del borde
    if y+h > rows: h = rows - y
    if x+w > cols: w = cols - x
    if y < 0: 
        fg = fg[-y:, :, :]
        h = fg.shape[0]
        y = 0
    if x < 0:
        fg = fg[:, -x:, :]
        w = fg.shape[1]
        x = 0

    # Canal alpha (transparencia)
    alpha = fg[:h, :w, 3] / 255.0

    # Mezclar cada canal de color
    for c in range(0, 3):
        bg[y:y+h, x:x+w, c] = alpha*fg[:h, :w, c] + (1-alpha)*bg[y:y+h, x:x+w, c]

    return bg

def analyze_threaded(frame_copy):
    """Analiza emociones en un thread separado para no bloquear el UI"""
    global last_emotions, is_analyzing
    
    try:
        results = DeepFace.analyze(
            frame_copy,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )

        if results and results[0]["face_confidence"] > FD_THRESHOLD:
            emotions_detected = []
            for res in results:
                weighted_scores = {k: float(res["emotion"][k] * WEIGHTS.get(k, 1.0)) for k in res["emotion"]}
                dominant_emotion = max(weighted_scores, key=weighted_scores.get)
                
                if LOGGING:
                    print(f"Detectado: {json.dumps(weighted_scores, indent=2)} \n -> {dominant_emotion} \n ------- \n")

                emotions_detected.append((res['region'], dominant_emotion))
            
            with emotion_lock:
                analysis_queue.put(emotions_detected)
        else:
            with emotion_lock:
                analysis_queue.put([])
                    
    except Exception as e:
        print("Error analizando emociones:", e)
        with emotion_lock:
            analysis_queue.put([])
    finally:
        is_analyzing = False

def trigger_flash(color, duration=0.3):
    """Activa un flash de pantalla completa"""
    global flash_color, flash_start
    flash_color = color
    flash_start = time.time()

def update_frame():
    global frame_count, last_emotions, start_time, current_emotion, level, dani_dice
    global detected, last_capture_time, is_analyzing, combo, max_combo, score
    global flash_color, flash_start, show_feedback, feedback_text, feedback_color

    # Intervalo entre capturas (segundos)
    CAPTURE_INTERVAL = 1.5

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame_count += 1
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (label.winfo_width(), label.winfo_height()))

    # Procesar resultados de análisis si están disponibles
    if not analysis_queue.empty():
        with emotion_lock:
            last_emotions = analysis_queue.get()
            if last_emotions and len(last_emotions) > 0:
                detected = EMOTIONS_TRANSLATED[last_emotions[0][1]]

    # Analizar solo cada CAPTURE_INTERVAL segundos en thread separado
    now = time.time()
    if now - last_capture_time >= CAPTURE_INTERVAL and not is_analyzing:
        last_capture_time = now
        is_analyzing = True
        frame_copy = frame.copy()
        Thread(target=analyze_threaded, args=(frame_copy,), daemon=True).start()

    # Mostrar temporizador
    time_left = int(TIME_TO_RESPOND - (time.time() - start_time))

    # Preparar texto de orden
    orden = ""
    if dani_dice:
        orden += "🎯 DANI DICE: "
    else:
        orden += "❌ DANI NO DICE: "
    orden += f"{current_emotion.upper()}"

    # Actualizar texto principal - MÁS GRANDE Y CLARO
    label_text.config(
        text=f"⏱️ {time_left}s  |  🏆 Nivel: {level}  |  🔥 Combo: {combo}x  |  ⭐ Score: {score}\n\n{orden}\n\n👤 Detectado: {detected.upper()}"
    )

    # Dibujar emociones en la pantalla con animación
    for region, emotion in last_emotions:
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        if emotion in loaded_emojis:
            emoji = loaded_emojis[emotion]
            # Efecto de "pop" - el emoji aparece más grande
            scale_factor = w/emoji.shape[1]
            frame = overlay_image(frame, emoji, x, y-20, scale=scale_factor)

    # Flash de pantalla para feedback visual
    if flash_color:
        elapsed = time.time() - flash_start
        if elapsed < 0.3:
            # Crear overlay semitransparente
            overlay = frame.copy()
            color_bgr = (0, 255, 0) if flash_color == 'green' else (0, 0, 255)
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color_bgr, -1)
            alpha = 0.3 * (1 - elapsed/0.3)  # fade out
            frame = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
        else:
            flash_color = None

    # Mostrar feedback temporal grande
    if show_feedback:
        text_size = cv2.getTextSize(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        
        # Sombra para mejor legibilidad
        cv2.putText(frame, feedback_text, (text_x+3, text_y+3), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)
        color = (0, 255, 0) if feedback_color == "green" else (0, 0, 255)
        cv2.putText(frame, feedback_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

    # Comprobar si debe cambiar emoción
    change_emotion = False
    tiempo_agotado = time.time() - start_time > TIME_TO_RESPOND
    emocion_correcta = detected == current_emotion

    # LÓGICA CORREGIDA
    if dani_dice:
        # DANI DICE: debes hacer la emoción
        if tiempo_agotado:
            print("⏰ Tiempo agotado. Reiniciando juego.")
            feedback_text = "⏰ TIEMPO AGOTADO!"
            feedback_color = "red"
            show_feedback = True
            trigger_flash('red')
            label_res.config(text=f"⏰ Tiempo agotado\n\n🏆 Nivel Alcanzado: {level}\n⭐ Score Final: {score}\n🔥 Mejor Combo: {max_combo}", fg="red")
            combo = 0
            score = 0
            level = 1
            change_emotion = True
            root.after(1500, lambda: setattr(globals(), 'show_feedback', False))
            
        elif emocion_correcta:
            print("✅ ¡Correcto!")
            combo += 1
            max_combo = max(max_combo, combo)
            score += 100 * combo
            feedback_text = f"✅ ¡CORRECTO! +{100*combo}"
            feedback_color = "green"
            show_feedback = True
            trigger_flash('green')
            label_res.config(text=f"✅ ¡Excelente!\n\n🔥 Combo: {combo}x\n⭐ +{100*combo} puntos", fg="green")
            level += 1
            change_emotion = True
            root.after(1500, lambda: setattr(globals(), 'show_feedback', False))
    else:
        # DANI NO DICE: NO debes hacer la emoción
        if tiempo_agotado:
            print("✅ ¡Correcto! No hiciste la emoción")
            combo += 1
            max_combo = max(max_combo, combo)
            score += 100 * combo
            feedback_text = f"✅ ¡BIEN! +{100*combo}"
            feedback_color = "green"
            show_feedback = True
            trigger_flash('green')
            label_res.config(text=f"✅ ¡Perfecto!\n\n🔥 Combo: {combo}x\n⭐ +{100*combo} puntos", fg="green")
            level += 1
            change_emotion = True
            root.after(1500, lambda: setattr(globals(), 'show_feedback', False))
            
        elif emocion_correcta:
            print("❌ ¡Te equivocaste! DANI NO LO DIJO")
            feedback_text = "❌ DANI NO LO DIJO!"
            feedback_color = "red"
            show_feedback = True
            trigger_flash('red')
            label_res.config(text=f"❌ DANI NO LO DIJO\n\n🏆 Nivel Alcanzado: {level}\n⭐ Score Final: {score}\n🔥 Mejor Combo: {max_combo}", fg="red")
            combo = 0
            score = 0
            level = 1
            change_emotion = True
            root.after(1500, lambda: setattr(globals(), 'show_feedback', False))

    if change_emotion:
        current_emotion, dani_dice = new_emotion(DS_THRESHOLD)
        start_time = time.time()
        detected = 'none'  # Reset detección

    # Mostrar en Tkinter
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    # Llamar de nuevo después de 10ms
    root.after(10, update_frame)

def actualizar_fondo(event):
    """Redimensiona el fondo al cambiar tamaño de ventana"""
    ancho = root.winfo_width()
    alto = root.winfo_height()
    
    if ancho > 1 and alto > 1:  # Evitar errores de redimensión
        imagen_redimensionada = imagen_original.resize((ancho, alto), Image.Resampling.LANCZOS)
        fondo_nuevo = ImageTk.PhotoImage(imagen_redimensionada)
        label_fondo.config(image=fondo_nuevo)
        label_fondo.image = fondo_nuevo
        
        label.configure(width=int(ancho*PROPORCION_VIDEO), height=int(alto*PROPORCION_VIDEO))

def pantalla_inicio():
    """Pantalla de inicio mejorada con botones más visibles"""
    path = os.path.join(IMG_PATH, "fondo.png")
    
    if not os.path.exists(path):
        print(f"Advertencia: No se encontró {path}")
        return 8  # Tiempo por defecto
    
    pantalla = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    pantalla = cv2.resize(pantalla, (ANCHO_VENTANA, ALTO_VENTANA))
    
    tiempo = 8
    selected = 0  # 0=fácil, 1=normal, 2=difícil
    tiempos = [8, 3, 1]
    nombres = ["FÁCIL", "NORMAL", "DIFÍCIL"]

    while True:
        display = pantalla.copy()
        y = 120
        
        # Título
        cv2.putText(display, "Bienvenido a DANI Dice", (int(ANCHO_VENTANA/3)+50, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        y += 60
        
        # Reglas
        for regla in REGLAS_TEXT[:-3]:  # Sin las instrucciones de teclado
            cv2.putText(display, regla, (int(ANCHO_VENTANA/3)+50, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y += 35

        # BOTONES GRANDES Y VISIBLES
        y += 30
        cv2.putText(display, "SELECCIONA DIFICULTAD:", (int(ANCHO_VENTANA/3)+50, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
        y += 60
        
        button_width = 300
        button_height = 80
        start_x = int(ANCHO_VENTANA/3) + 50
        
        for i, (nombre, tiempo_val) in enumerate(zip(nombres, tiempos)):
            color = (100, 255, 100) if i == selected else (200, 200, 200)
            cv2.rectangle(display, (start_x, y), (start_x+button_width, y+button_height), color, -1)
            cv2.rectangle(display, (start_x, y), (start_x+button_width, y+button_height), (0, 0, 0), 3)
            
            text = f"{i+1}. {nombre} ({tiempo_val}s)"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = start_x + (button_width - text_size[0]) // 2
            text_y = y + (button_height + text_size[1]) // 2
            cv2.putText(display, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            y += button_height + 20

        y += 30
        cv2.putText(display, "Presiona 1, 2 o 3 para comenzar | ESC para salir", 
                    (int(ANCHO_VENTANA/3)+50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("DANI Dice - Inicio", display)
        key = cv2.waitKey(100) & 0xFF

        if key == ord('1'):
            cv2.destroyAllWindows()
            return tiempos[0]
        elif key == ord('2'):
            cv2.destroyAllWindows()
            return tiempos[1]
        elif key == ord('3'):
            cv2.destroyAllWindows()
            return tiempos[2]
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            exit()
        
        # Animación de selección
        selected = (selected + 1) % 3 if frame_count % 20 == 0 else selected

def new_emotion(DS_THRESHOLD):
    """Genera nueva emoción y decide si 'Dani dice'"""
    current_emotion = random.choice(EMOTIONS)
    prob_dani_dice = random.random()
    dani_dice = DS_THRESHOLD > prob_dani_dice
    
    return current_emotion, dani_dice

# ====== INICIO DEL PROGRAMA ======

TIME_TO_RESPOND = pantalla_inicio()
print(f"⏱️ Tiempo para responder: {TIME_TO_RESPOND} segundos")

# Configuración cámara con manejo de errores
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: No se pudo acceder a la cámara")
    messagebox.showerror("Error de Cámara", 
                        "No se detectó ninguna cámara.\n\nAsegúrate de que:\n"
                        "- La cámara esté conectada\n"
                        "- Ninguna otra aplicación la esté usando\n"
                        "- Tengas permisos para acceder a la cámara")
    exit()

# Configurar resolución para mejor rendimiento
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Cargar emojis
loaded_emojis = {}
for emotion, file in EMOTION_SPRITES.items():
    path = os.path.join(IMG_PATH, file)
    if os.path.exists(path):
        loaded_emojis[emotion] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        print(f"⚠️ Advertencia: No se encontró {path}")

# Inicializar variables del juego
current_emotion, dani_dice = new_emotion(DS_THRESHOLD)
start_time = time.time()

# ====== VENTANA TKINTER ======

root = tk.Tk()
root.title("🎮 DANI DICE - DETECCIÓN DE EMOCIONES")
root.geometry(f"{ANCHO_VENTANA}x{ALTO_VENTANA}")

# Fondo
ruta_fondo = os.path.join(IMG_PATH, "fondo.png")
if os.path.exists(ruta_fondo):
    imagen_original = Image.open(ruta_fondo)
    fondo_tk = ImageTk.PhotoImage(imagen_original)
else:
    # Crear fondo simple si no existe
    imagen_original = Image.new('RGB', (ANCHO_VENTANA, ALTO_VENTANA), (240, 240, 250))

label_fondo = tk.Label(root)
label_fondo.place(x=0, y=0, relwidth=1, relheight=1)

# Texto principal - MÁS GRANDE
label_text = tk.Label(
    label_fondo, 
    text="", 
    font=("Arial", 20, "bold"), 
    fg="black",
    bg="white",
    justify="center",
    padx=20,
    pady=10
)
label_text.place(relx=0.5, rely=0.08, anchor="n")

# Texto de resultados - MÁS VISIBLE
label_res = tk.Label(
    label_fondo, 
    text="", 
    font=("Arial", 28, "bold"), 
    fg="green",
    bg="white",
    justify="center",
    padx=30,
    pady=20
)
label_res.place(relx=0.95, rely=0.5, anchor="e")

# Frame del video
frame_display = tk.Frame(root, bg="black", highlightthickness=3, highlightbackground="purple")
frame_display.place(relx=0.5, rely=0.5, anchor="center")

label = tk.Label(frame_display, bg="black")
label.pack()

# High Score Display
label_highscore = tk.Label(
    label_fondo,
    text=f"🏆 Mejor Score del Día: {high_score}",
    font=("Arial", 16, "bold"),
    fg="gold",
    bg="purple",
    padx=20,
    pady=5
)
label_highscore.place(relx=0.5, rely=0.95, anchor="s")

root.bind("<Configure>", actualizar_fondo)

update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()