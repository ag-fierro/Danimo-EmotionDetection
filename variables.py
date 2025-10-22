LOGGING = True  # activar/desactivar logs de depuración

IMG_PATH = "src/img/"

FD_THRESHOLD = 0.7  # umbral de confianza para detección facial
DS_THRESHOLD = 0.7  # probabilidad de que "DANI diga"

FRAME_INTERVAL = 7  # analizar 1 de cada X frames

ANCHO_VENTANA = 1600
ALTO_VENTANA = 900

PROPORCION_VIDEO = 0.50

SLEEP_TIME = 5  # tiempo de espera tras cada ronda

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
    'disgust': 2.0,
    'fear': 1.2,
    'happy': 1.0,
    'sad': 1.0,
    'surprise': 2.0,
    'neutral': 1.2
}

REGLAS_TEXT = [
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