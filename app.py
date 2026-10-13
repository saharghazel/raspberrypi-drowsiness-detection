"""Raspberry Pi 4 drowsiness detection application.

USB camera (V4L2) -> OpenCV Haar eye detection -> existing TFLite model
-> Flask dashboard and optional GPIO alerts.
"""
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import threading
import time
import collections
import os
from pathlib import Path

# Lightweight TFLite runtime is preferable on Raspberry Pi when available.
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

try:
    from gpiozero import LED, Buzzer
except ImportError:
    LED = Buzzer = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_yeux.tflite"
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
GPIO_ALERTS = os.getenv("GPIO_ALERTS", "0") == "1"
LED_PIN = int(os.getenv("LED_PIN", "17"))
BUZZER_PIN = int(os.getenv("BUZZER_PIN", "27"))
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "5000"))
led = buzzer = None
if GPIO_ALERTS:
    if LED is None:
        print("GPIO warning: gpiozero not installed; dashboard alerts remain available")
    else:
        try:
            led = LED(LED_PIN)
            buzzer = Buzzer(BUZZER_PIN)
        except Exception as exc:
            print(f"GPIO unavailable: {exc}. Dashboard alerts remain available")
            for device in (led, buzzer):
                if device is not None:
                    device.close()
            led = buzzer = None

app = Flask(__name__)

# ── Supprimer les warnings TensorFlow inutiles ────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ── Charger le modèle TFLite ──────────────────────────────────────────
print("Chargement du modele TFLite...")
interpreter = Interpreter(model_path=str(MODEL_PATH), num_threads=2)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"Modele charge ! Input shape : {input_details[0]['shape']}")
# Doit afficher : [ 1 64 64 1]

# ── Charger les classifieurs Haar (inclus dans OpenCV, pas d'install) ─
# Ces fichiers XML sont dans le dossier d'installation d'OpenCV
cascade_visage = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
cascade_oeil = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

if cascade_visage.empty() or cascade_oeil.empty():
    print("ERREUR : Fichiers Haar introuvables !")
    print("Réinstallez opencv-python : pip install --upgrade opencv-python")
else:
    print("Classifieurs Haar charges !")

# ── Variables globales ─────────────────────────────────────────────────
etat = {
    "label"      : "En attente...",
    "score"      : 0.0,
    "alerte"     : False,
    "compteur"   : 0,
    "yeux_ok"    : False,
    "nb_yeux"    : 0,
    "historique" : collections.deque(maxlen=50)
}
lock = threading.Lock()

# Frame partagée entre le thread d'inférence et le générateur vidéo
frame_courante = None
frame_lock     = threading.Lock()

# ── Ouvrir la webcam ───────────────────────────────────────────────────
print("Ouverture de la webcam...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    print("ERREUR : Webcam introuvable ! Verify /dev/video0 or set CAMERA_INDEX=1")
else:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("Webcam ouverte !")

# ── Paramètres ─────────────────────────────────────────────────────────
SEUIL_CNN  = 0.55   # Probabilité > 55% → sleepy
ALERTE_N   = 10     # Nb frames sleepy consécutives → alerte
IMG_SIZE   = 64     # IDENTIQUE à l'entraînement

# ── Prétraitement IDENTIQUE à l'entraînement MRL ──────────────────────
def pretraiter_oeil(region_bgr):
    """
    Pipeline identique à flow_from_directory :
      1. Grayscale  (color_mode='grayscale')
      2. Resize 64x64  (target_size=(64,64))
      3. /255.0  (rescale=1/255)
      4. shape [1, 64, 64, 1]
    """
    if region_bgr is None or region_bgr.size == 0:
        return None
    # 1. Grayscale
    if len(region_bgr.shape) == 3:
        gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = region_bgr
    # 2. Resize
    img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    # 3. Normaliser
    arr = img.astype(np.float32) / 255.0
    # 4. Shape correcte
    return arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def predire_oeil(region_bgr):
    """Retourne prob sleepy (0.0=awake, 1.0=sleepy)."""
    arr = pretraiter_oeil(region_bgr)
    if arr is None:
        return 0.0
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    return float(interpreter.get_tensor(output_details[0]['index'])[0][0])


# ── Thread de capture + inférence ─────────────────────────────────────
def boucle_capture():
    global frame_courante
    score_courant = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Détecter le visage ──────────────────────────────────
        visages = cascade_visage.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        yeux_detectes = []
        score_courant = 0.0

        if len(visages) > 0:
            # Prendre le plus grand visage
            visages = sorted(visages, key=lambda v: v[2]*v[3], reverse=True)
            (vx, vy, vw, vh) = visages[0]

            # Chercher les yeux DANS la région du visage (moitié supérieure)
            roi_gray = gray_frame[vy:vy+vh//2, vx:vx+vw]
            roi_bgr  = frame[vy:vy+vh//2, vx:vx+vw]

            yeux = cascade_oeil.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )

            scores = []
            for (ex, ey, ew, eh) in yeux[:2]:  # Max 2 yeux
                # Extraire la région de l'œil
                oeil_bgr = roi_bgr[ey:ey+eh, ex:ex+ew]
                prob     = predire_oeil(oeil_bgr)
                scores.append(prob)
                yeux_detectes.append((vx+ex, vy+ey, ew, eh, prob))

            if scores:
                score_courant = sum(scores) / len(scores)

        # ── Décision ────────────────────────────────────────────
        yeux_ok  = len(yeux_detectes) > 0
        sleepy   = yeux_ok and (score_courant > SEUIL_CNN)
        label    = "SLEEPY" if sleepy else ("AWAKE" if yeux_ok else "En attente...")

        # ── Mettre à jour l'état global ─────────────────────────
        with lock:
            etat["score"]   = round(score_courant * 100, 1)
            etat["label"]   = label
            etat["yeux_ok"] = yeux_ok
            etat["nb_yeux"] = len(yeux_detectes)
            etat["historique"].append({
                "t"     : time.strftime("%H:%M:%S"),
                "score" : round(score_courant * 100, 1),
                "etat"  : "sleepy" if sleepy else "awake",
                "yeux"  : len(yeux_detectes)
            })
            if sleepy:
                etat["compteur"] += 1
                etat["alerte"]    = etat["compteur"] >= ALERTE_N
            else:
                if not yeux_ok or score_courant < (SEUIL_CNN - 0.1):
                    etat["compteur"] = 0
                    etat["alerte"]   = False

        # Optional Raspberry Pi GPIO outputs, synchronized with the alert.
        if led is not None and buzzer is not None:
            with lock:
                gpio_alert = etat["alerte"]
            if gpio_alert:
                led.on()
                buzzer.on()
            else:
                led.off()
                buzzer.off()

        # ── Annoter le frame ────────────────────────────────────
        frame_annotee = frame.copy()

        # Dessiner le visage détecté
        if len(visages) > 0:
            (vx, vy, vw, vh) = visages[0]
            cv2.rectangle(frame_annotee, (vx, vy), (vx+vw, vy+vh),
                          (100, 100, 200), 2)

        # Dessiner les yeux avec leur score
        for (ex, ey, ew, eh, prob) in yeux_detectes:
            col = (0, 80, 220) if prob > SEUIL_CNN else (30, 200, 30)
            cv2.rectangle(frame_annotee, (ex, ey), (ex+ew, ey+eh), col, 2)
            cv2.putText(frame_annotee, f"{prob*100:.0f}%",
                        (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col, 1)

        # Bandeau infos en haut
        cv2.rectangle(frame_annotee, (0, 0), (w, 65), (15, 15, 25), -1)

        if yeux_ok:
            isSleepy = sleepy
            couleur  = (0, 80, 220) if isSleepy else (30, 200, 30)
            txt = f"{'SLEEPY' if isSleepy else 'AWAKE'}  CNN:{score_courant*100:.0f}%  Yeux:{len(yeux_detectes)}"
            cv2.putText(frame_annotee, txt, (10, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, couleur, 2)
        else:
            cv2.putText(frame_annotee, "Aucun visage / yeux detectes",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (140, 140, 140), 2)

        # Alerte
        with lock:
            alerte_on = etat["alerte"]
            compteur  = etat["compteur"]

        cv2.putText(frame_annotee,
                    f"Alerte dans : {max(0, ALERTE_N - compteur)} frames",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (100, 100, 160), 1)

        if alerte_on:
            cv2.rectangle(frame_annotee, (0, 0), (w-1, h-1), (0, 0, 255), 10)
            cv2.rectangle(frame_annotee, (0, h-50), (w, h), (0, 0, 180), -1)
            cv2.putText(frame_annotee,
                        "!!! SOMNOLENCE DETECTEE !!!",
                        (w//2 - 200, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        (255, 255, 255), 2)

        # Sauvegarder le frame annoté
        with frame_lock:
            frame_courante = frame_annotee.copy()

        time.sleep(0.1)   # ~10 analyses/sec


# Démarrer le thread
t = threading.Thread(target=boucle_capture, daemon=True)
t.start()
print("Thread d'inference demarre !")


# ── Générateur flux vidéo ──────────────────────────────────────────────
def generer_flux():
    while True:
        with frame_lock:
            frame = None if frame_courante is None else frame_courante.copy()
        if frame is None:
            time.sleep(0.05)
            continue

        ok, buf = cv2.imencode('.jpg', frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes()
                   + b'\r\n')
        time.sleep(0.033)   # ~30 fps flux vidéo


# ── Routes Flask ───────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video')
def video():
    return Response(generer_flux(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/etat')
def api_etat():
    with lock:
        return jsonify({
            "label"     : etat["label"],
            "score"     : etat["score"],
            "alerte"    : etat["alerte"],
            "compteur"  : etat["compteur"],
            "yeux_ok"   : etat["yeux_ok"],
            "nb_yeux"   : etat["nb_yeux"],
            "historique": list(etat["historique"])[-15:]
        })

# ── Démarrage ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Dashboard demarre !")
    print("  Ouvre ton navigateur :")
    print(f"  http://{HOST}:{PORT}")
    print("="*50 + "\n")
    app.run(debug=False, host=HOST, port=PORT, threaded=True, use_reloader=False)
