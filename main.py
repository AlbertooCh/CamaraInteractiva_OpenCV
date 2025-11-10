import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque

# --- Configuración Inicial ---

# Configuración de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Configuración de OpenCV
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

# Cargar el Clasificador de Caras de OpenCV
try:
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise IOError("No se pudo cargar el clasificador de caras")
    print("Clasificador de caras cargado exitosamente.")
except Exception as e:
    print(f"Error cargando Haar Cascade: {e}")
    print("La detección de caras estará deshabilitada.")
    face_cascade = None

# Variables de estado
zoom_level = 1.0
is_recording = False
video_writer = None
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 20.0

# Variables de Zoom Relativo
zoom_gesture_active = False
initial_zoom_distance = 0.0
initial_zoom_level = 1.0
MIN_ZOOM = 1.0
MAX_ZOOM = 4.0
ZOOM_SENSITIVITY = 5.0

# Variables de estado para Filtros y Efectos
filters = ["NORMAL", "GRISES", "BORDES", "BLUR", "VERDE_HSV", "COMIC"]
current_filter_index = 0
show_face_detection = False
show_feature_points = False

# Variables de Brillo y Contraste (OpenCV puro)
brightness_control_active = False
current_alpha = 1.0  # Contraste (1.0 = normal)
current_beta = 0.0  # Brillo (0 = normal)

# Constantes de Tiempo
START_RECORD_HOLD_SECONDS = 1.0
STOP_RECORD_HOLD_SECONDS = 1.0
PHOTO_HOLD_SECONDS = 2.0
PHOTO_COUNTDOWN_SECONDS = 3.0
VIDEO_TRIM_SECONDS = 1.5
GESTURE_COOLDOWN = 1.5

# Búfer de Vídeo
video_buffer = deque(maxlen=int(fps * VIDEO_TRIM_SECONDS))

# Variables de estado para Gestos
photo_gesture_start_time = 0
start_record_gesture_start_time = 0
stop_record_gesture_start_time = 0
recording_start_time = 0
photo_countdown_start_time = 0
toggle_record_scheduled_time = 0
scheduled_frame_for_action = None

# Timers de Cooldown para nuevos gestos
last_filter_toggle_time = 3
last_face_toggle_time = 3
last_feature_toggle_time = 3

# Variables para mensaje de LÍMITE
show_limit_message = False
limit_message_time = 0

print("Iniciando cámara... Presiona 'q' para salir.")
print("--- MODO DE USO ---")
print("- 1 Mano (1s): Puño (Iniciar Grabación)")
print("- 1 Mano (2s): Paz (Foto)")
print("- 1 Mano (tocar): Señalar 👆 (Cambiar Filtro)")
print("- 1 Mano (tocar): Gesto OK 👌 (Activar Detección de Caras)")
print("- 1 Mano (tocar): Gesto 'Tres Dedos' 🤟 (Mostrar Puntos de la Mano)")
print("- 2 Manos (Zoom): Puño + Mano Abierta")
print("- 2 Manos (1s): Dos Puños (Parar Grabación)")
print("- 2 Manos (sostener): Señalar 👆👆 (Control Brillo/Contraste)")
print("---")


# --- Funciones de Detección de Gestos ---

def get_landmark_distance(landmarks, idx1, idx2):
    if landmarks:
        lm1 = landmarks.landmark[idx1]
        lm2 = landmarks.landmark[idx2]
        return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)
    return 0


def get_hand_center(landmarks):
    return landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]


## Señal de paz: índice y medio arriba, anular y meñique abajo
def is_peace_sign(landmarks):
    if not landmarks: return False
    index_up = landmarks.landmark[8].y < landmarks.landmark[6].y
    middle_up = landmarks.landmark[12].y < landmarks.landmark[10].y
    ring_down = landmarks.landmark[16].y > landmarks.landmark[14].y
    pinky_down = landmarks.landmark[20].y > landmarks.landmark[18].y
    return index_up and middle_up and ring_down and pinky_down

## Señal de puño: todos los dedos cerrados
def is_fist(landmarks):
    if not landmarks: return False
    index_closed = landmarks.landmark[8].y > landmarks.landmark[6].y
    middle_closed = landmarks.landmark[12].y > landmarks.landmark[10].y
    ring_closed = landmarks.landmark[16].y > landmarks.landmark[14].y
    pinky_closed = landmarks.landmark[20].y > landmarks.landmark[18].y
    return index_closed and middle_closed and ring_closed and pinky_closed

## Señal de mano abierta: todos los dedos arriba
def is_open_hand(landmarks):
    if not landmarks: return False
    index_up = landmarks.landmark[8].y < landmarks.landmark[6].y
    middle_up = landmarks.landmark[12].y < landmarks.landmark[10].y
    ring_up = landmarks.landmark[16].y < landmarks.landmark[14].y
    pinky_up = landmarks.landmark[20].y < landmarks.landmark[18].y
    thumb_up = landmarks.landmark[4].y < landmarks.landmark[3].y
    return index_up and middle_up and ring_up and pinky_up and thumb_up

## Señal de señalar hacia arriba: índice arriba, otros dedos abajo
def is_pointing_up(landmarks):
    if not landmarks: return False
    index_up = landmarks.landmark[8].y < landmarks.landmark[6].y
    middle_down = landmarks.landmark[12].y > landmarks.landmark[10].y
    ring_down = landmarks.landmark[16].y > landmarks.landmark[14].y
    pinky_down = landmarks.landmark[20].y > landmarks.landmark[18].y
    return index_up and middle_down and ring_down and pinky_down

## Señal de OK: pulgar e índice juntos, otros dedos arriba
def is_ok_sign(landmarks):
    if not landmarks: return False
    pinch_distance = get_landmark_distance(landmarks, 4, 8)
    is_pinching = pinch_distance < 0.1
    middle_up = landmarks.landmark[12].y < landmarks.landmark[10].y
    ring_up = landmarks.landmark[16].y < landmarks.landmark[14].y
    pinky_up = landmarks.landmark[20].y < landmarks.landmark[18].y
    return is_pinching and middle_up and ring_up and pinky_up

## Señal de 'Tres Dedos': índice, medio, anular arriba; meñique cerrado; pulgar doblado
def is_three_fingers_up(landmarks):
    """Verifica Gesto '3 dedos' (índice, medio, anular)."""
    if not landmarks: return False
    index_up = landmarks.landmark[8].y < landmarks.landmark[6].y
    middle_up = landmarks.landmark[12].y < landmarks.landmark[10].y
    ring_up = landmarks.landmark[16].y < landmarks.landmark[14].y
    # Meñique cerrado
    pinky_down = landmarks.landmark[20].y > landmarks.landmark[18].y
    # Pulgar 'doblado' hacia adentro (comparación en X)
    thumb_in = landmarks.landmark[4].x > landmarks.landmark[3].x
    return index_up and middle_up and ring_up and pinky_down and thumb_in



def apply_zoom(frame, zoom):
    if zoom == 1.0:
        return frame
    h, w, _ = frame.shape
    crop_w = int(w / zoom)
    crop_h = int(h / zoom)
    x = int((w - crop_w) / 2)
    y = int((h - crop_h) / 2)
    zoomed_frame = frame[max(0, y):min(h, y + crop_h), max(0, x):min(w, x + crop_w)]
    return cv2.resize(zoomed_frame, (w, h), interpolation=cv2.INTER_LINEAR)


# --- Bucle Principal ---

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Esperando frame...")
        time.sleep(0.1)
        continue

    frame = cv2.flip(frame, 1)

    # --- APLICAR BRILLO/CONTRASTE PRIMERO ---
    processed_frame = cv2.convertScaleAbs(frame, alpha=current_alpha, beta=current_beta)

    # Usar el frame original (sin brillo) para la detección de gestos
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    current_time = time.time()

    num_hands = 0
    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)

    if photo_countdown_start_time > 0:
        num_hands = 0

    if num_hands == 2:
        hand1_lm = results.multi_hand_landmarks[0]
        hand2_lm = results.multi_hand_landmarks[1]
        hand1_fist = is_fist(hand1_lm)
        hand2_fist = is_fist(hand2_lm)
        hand1_pointing = is_pointing_up(hand1_lm)
        hand2_pointing = is_pointing_up(hand2_lm)

        if hand1_pointing and hand2_pointing:
            brightness_control_active = True
            current_beta = np.interp(hand1_lm.landmark[8].y, [0.1, 0.9], [100.0, -100.0])
            current_alpha = np.interp(hand2_lm.landmark[8].y, [0.1, 0.9], [3.0, 0.5])
            zoom_gesture_active = False
            stop_record_gesture_start_time = 0
        elif is_recording and hand1_fist and hand2_fist:
            if stop_record_gesture_start_time == 0:
                stop_record_gesture_start_time = current_time
            elif (current_time - stop_record_gesture_start_time) > STOP_RECORD_HOLD_SECONDS:
                if toggle_record_scheduled_time == 0:
                    toggle_record_scheduled_time = current_time
                    print("Parada de grabación programada.")
                stop_record_gesture_start_time = 0
            zoom_gesture_active = False
            brightness_control_active = False
        elif (hand1_fist and is_open_hand(hand2_lm)) or (is_open_hand(hand1_lm) and hand2_fist):
            center1 = get_hand_center(hand1_lm)
            center2 = get_hand_center(hand2_lm)
            current_distance = math.hypot(center1.x - center2.x, center1.y - center2.y)
            if not zoom_gesture_active:
                zoom_gesture_active = True
                initial_zoom_distance = current_distance
                initial_zoom_level = zoom_level
            else:
                distance_change = current_distance - initial_zoom_distance
                new_zoom = initial_zoom_level - (distance_change * ZOOM_SENSITIVITY)
                if new_zoom > MAX_ZOOM:
                    zoom_level, show_limit_message, limit_message_time = MAX_ZOOM, True, current_time
                elif new_zoom < MIN_ZOOM:
                    zoom_level, show_limit_message, limit_message_time = MIN_ZOOM, True, current_time
                else:
                    zoom_level, show_limit_message = new_zoom, False
            photo_gesture_start_time = 0
            start_record_gesture_start_time = 0
            brightness_control_active = False
        else:
            zoom_gesture_active = False
            stop_record_gesture_start_time = 0
            brightness_control_active = False

    elif num_hands == 1:
        # Lógica de 1 mano
        zoom_gesture_active = False
        stop_record_gesture_start_time = 0
        brightness_control_active = False
        hand_landmarks = results.multi_hand_landmarks[0]

        if is_peace_sign(hand_landmarks):
            if photo_gesture_start_time == 0:
                photo_gesture_start_time = current_time
            elif (current_time - photo_gesture_start_time) > PHOTO_HOLD_SECONDS:
                if photo_countdown_start_time == 0:
                    photo_countdown_start_time = current_time
                    print(f"Iniciando cuenta atrás de {PHOTO_COUNTDOWN_SECONDS}s para la foto.")
                photo_gesture_start_time = 0
        else:
            photo_gesture_start_time = 0

        if is_fist(hand_landmarks) and not is_recording:
            if start_record_gesture_start_time == 0:
                start_record_gesture_start_time = current_time
            elif (current_time - start_record_gesture_start_time) > START_RECORD_HOLD_SECONDS:
                if toggle_record_scheduled_time == 0:
                    toggle_record_scheduled_time = current_time
                    print("Inicio de grabación programado.")
                start_record_gesture_start_time = 0
        else:
            start_record_gesture_start_time = 0

        if is_pointing_up(hand_landmarks) and (current_time - last_filter_toggle_time > GESTURE_COOLDOWN):
            current_filter_index = (current_filter_index + 1) % len(filters)
            print(f"Cambiando a filtro: {filters[current_filter_index]}")
            last_filter_toggle_time = current_time
            photo_gesture_start_time = 0
            start_record_gesture_start_time = 0

        if is_ok_sign(hand_landmarks) and (current_time - last_face_toggle_time > GESTURE_COOLDOWN):
            if face_cascade:
                show_face_detection = not show_face_detection
                print(f"Detección de caras: {'Activada' if show_face_detection else 'Desactivada'}")
            else:
                print("Haar Cascade no cargado.")
            last_face_toggle_time = current_time
            photo_gesture_start_time = 0
            start_record_gesture_start_time = 0

        if is_three_fingers_up(hand_landmarks) and (current_time - last_feature_toggle_time > GESTURE_COOLDOWN):
            show_feature_points = not show_feature_points
            print(f"Mostrar puntos de la mano: {'Activado' if show_feature_points else 'Desactivado'}")
            last_feature_toggle_time = current_time
            photo_gesture_start_time = 0
            start_record_gesture_start_time = 0

    else:  # 0 manos
        zoom_gesture_active = False
        photo_gesture_start_time = 0
        start_record_gesture_start_time = 0
        stop_record_gesture_start_time = 0
        brightness_control_active = False

    # --- ################################################################# ---
    # --- SECCIÓN DE PROCESAMIENTO DE IMAGEN (OpenCV) ---
    # --- ################################################################# ---
    faces = []

    gray_for_detection = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detect_faces_now = (show_face_detection or filters[current_filter_index] == "BLUR") and face_cascade
    if detect_faces_now:
        faces = face_cascade.detectMultiScale(
            gray_for_detection, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

    filter_name = filters[current_filter_index]

    if filter_name == "GRISES":
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    elif filter_name == "BORDES":
        gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.Canny(gray_frame, 100, 200)
    elif filter_name == "BLUR":
        processed_frame = cv2.GaussianBlur(processed_frame, (21, 21), 0)
        for (x, y, w, h) in faces:
            sharp_face = frame[y:y + h, x:x + w]
            processed_frame[y:y + h, x:x + w] = cv2.convertScaleAbs(sharp_face, alpha=current_alpha, beta=current_beta)
    elif filter_name == "VERDE_HSV":
        hsv_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        processed_frame = cv2.bitwise_and(processed_frame, processed_frame, mask=mask)
    elif filter_name == "COMIC":
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 9)
        processed_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # 3. Dibujar Overlays
    if len(processed_frame.shape) == 2:
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

    # Dibujar el esqueleto de la mano (si está activado)
    if show_feature_points and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                processed_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),  # Puntos (Cyan)
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # Conexiones (Verde)
            )

    # Dibujar caras (si está activado)
    if show_face_detection:
        for (x, y, w, h) in faces:
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # --- Fin de la sección de procesamiento ---
    # --- ################################################################# ---

    # Aplicar zoom al frame procesado
    frame_for_action = apply_zoom(processed_frame, zoom_level)

    # Añadir el frame final (con filtros y zoom) al búfer de vídeo
    video_buffer.append(frame_for_action.copy())

    # --- Procesar acciones programadas (Foto, Grabar) ---

    if photo_countdown_start_time > 0:
        elapsed_countdown = current_time - photo_countdown_start_time
        seconds_left = PHOTO_COUNTDOWN_SECONDS - elapsed_countdown
        if seconds_left > 0:
            countdown_text = f"{int(math.ceil(seconds_left))}"
            text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 4, 10)[0]
            text_x = (frame_width - text_size[0]) // 2
            text_y = (frame_height + text_size[1]) // 2
            cv2.putText(frame_for_action, countdown_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 10)
        else:
            photo_frame = video_buffer[-1]
            filename = f"foto_{int(current_time)}.png"
            cv2.imwrite(filename, photo_frame)
            print(f"¡Foto guardada! {filename}")
            photo_countdown_start_time = 0
            cv2.putText(frame_for_action, "FOTO GUARDADA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    if toggle_record_scheduled_time != 0 and current_time >= toggle_record_scheduled_time:
        if not is_recording:
            is_recording = True
            recording_start_time = current_time
            filename = f"video_{int(current_time)}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
            video_buffer.clear()
            cv2.putText(frame_for_action, "GRABACION INICIADA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            is_recording = False
            if video_writer:
                video_writer.release()
                video_writer = None
                video_buffer.clear()
                print("Grabacion detenida y guardada (últimos 1.5s recortados).")
            cv2.putText(frame_for_action, "GRABACION DETENIDA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        toggle_record_scheduled_time = 0

    # --- Acciones de OpenCV (visualización y grabación) ---

    display_frame = frame_for_action

    if is_recording and video_writer:
        if len(video_buffer) >= video_buffer.maxlen:
            video_writer.write(video_buffer.popleft())
        elapsed_seconds = int(current_time - recording_start_time)
        minutes = elapsed_seconds // 60
        seconds = elapsed_seconds % 60
        timer_text = f"REC: {minutes:02d}:{seconds:02d}"
        cv2.putText(display_frame, timer_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- Feedback visual de gestos y límites ---
    if photo_gesture_start_time != 0:
        cv2.putText(display_frame, "Sostenga para foto...", (50, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
    elif start_record_gesture_start_time != 0:
        cv2.putText(display_frame, "Sostenga para REC...", (50, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
    elif stop_record_gesture_start_time != 0:
        cv2.putText(display_frame, "Sostenga para PARAR...", (50, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)

    if show_limit_message:
        if (current_time - limit_message_time) < 1.0:
            cv2.putText(display_frame, "ZOOM LIMITE", (frame_width // 2 - 100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            show_limit_message = False

    # --- Feedback de estado de OpenCV ---
    y_pos_feed = frame_height - 20

    cv2.putText(display_frame, f"Filtro: {filters[current_filter_index]}", (20, y_pos_feed),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if show_face_detection:
        cv2.putText(display_frame, "Caras: ON", (200, y_pos_feed),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if show_feature_points:
        cv2.putText(display_frame, "Mano Puntos: ON", (350, y_pos_feed),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Color Cyan/Verde

    # Feedback de Brillo/Contraste
    if brightness_control_active:
        cv2.putText(display_frame, f"Brillo (Izq): {current_beta:.0f}", (frame_width // 2 - 100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display_frame, f"Contraste (Der): {current_alpha:.1f}", (frame_width // 2 - 100, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        cv2.putText(display_frame, f"Zoom: {zoom_level:.1f}x", (frame_width - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2)

    cv2.imshow('Camara Interactiva - Gestos OpenCV', display_frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Limpieza Final ---
print("Cerrando todo...")
cap.release()
if video_writer:
    video_writer.release()
video_buffer.clear()
cv2.destroyAllWindows()
hands.close()