"""
=============================================================================
SISTEMA DE LOCK VISUAL BIOMÉTRICO CON DASHBOARD DE WEBCAMS
Proyecto: Lock Visual / Control de Acceso Facial
Autor: Pablo Ticeran
=============================================================================
Este módulo integra:
1. Dashboard inteligente de detección y selección de webcams:
   - Filtra cámaras virtuales inactivas (OBS, NVIDIA Broadcast, DroidCam sin señal).
   - Genera miniaturas visuales en tiempo real de cada dispositivo funcional.
   - Permite seleccionar la cámara mediante clics del ratón o teclado.
2. Inferencia y Reconocimiento Facial en tiempo real:
   - Compara los vectores biométricos contra 'modelo_biometrico.pkl'.
   - Muestra el HUD de estado ("DESBLOQUEADO" o "LOCK: NO AUTORIZADO").
   - Permite cambiar de cámara en caliente presionando la tecla 'c'.
=============================================================================
"""

import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import sys
import time
import pickle
import argparse
import cv2
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass
import numpy as np
import face_recognition

# Intentar importar pygrabber para obtener los nombres reales de DirectShow en Windows
try:
    from pygrabber.dshow_graph import FilterGraph
    HAS_PYGRABBER = True
except ImportError:
    HAS_PYGRABBER = False


def resolve_path(target_path):
    """Resuelve la ruta relativa al directorio de trabajo o a la raíz del proyecto."""
    if os.path.isabs(target_path) and os.path.exists(target_path):
        return target_path
    if os.path.exists(target_path):
        return os.path.abspath(target_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    root_relative = os.path.join(project_root, target_path)
    if os.path.exists(root_relative):
        return root_relative
    return os.path.abspath(target_path)


def load_biometric_model(model_path="modelo_biometrico.pkl"):
    """Carga y valida el archivo binario del modelo serializado."""
    model_resolved = resolve_path(model_path)
    if not os.path.exists(model_resolved):
        raise FileNotFoundError(
            f"No se encontró el modelo en '{model_resolved}'. "
            "Asegúrate de ejecutar primero 'MODEL_TRAINER/model_trainer.py'."
        )
    print(f"[*] Cargando modelo biométrico desde: {model_resolved}")
    with open(model_resolved, "rb") as f:
        data = pickle.load(f)
    print(f"[*] Identidades registradas: {data.get('classes', list(set(data['names'])))}")
    print(f"[*] Total de descriptores cargados: {len(data['encodings'])}\n")
    return data


# =============================================================================
# MÓDULO DE ESCANEO Y DASHBOARD DE WEBCAMS
# =============================================================================

def scan_available_webcams(max_probe=6):
    """
    Escanea los dispositivos de video disponibles en el sistema.
    Detecta resolución, nombres de dispositivo y si están enviando señal real
    o si son pantallas negras/cámaras virtuales inactivas de otros programas.
    """
    device_names = []
    if HAS_PYGRABBER:
        try:
            device_names = FilterGraph().get_input_devices()
        except Exception:
            device_names = []

    total_to_check = len(device_names) if device_names else max_probe
    cams = []

    print("\n" + "=" * 80)
    print("      ESCANEANDO DISPOSITIVOS DE VIDEO Y WEBCAMS DEL SISTEMA...")
    print("=" * 80)

    for i in range(total_to_check):
        name = device_names[i] if i < len(device_names) else f"Camara Index {i}"

        # Usar backend DirectShow en Windows para máxima compatibilidad con FilterGraph
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(i, backend)

        if not cap.isOpened():
            cap.release()
            cams.append({
                "index": i,
                "name": name,
                "opened": False,
                "has_frame": False,
                "status": "NO DISPONIBLE / CERRADA",
                "status_code": "OFFLINE",
                "res": "---",
                "mean_brightness": 0.0,
                "thumb": None
            })
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            cams.append({
                "index": i,
                "name": name,
                "opened": True,
                "has_frame": False,
                "status": "SIN FOTOGRAMA",
                "status_code": "NO_FEED",
                "res": "---",
                "mean_brightness": 0.0,
                "thumb": None
            })
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mean_val = float(np.mean(frame))
        std_val = float(np.std(frame))
        
        # Analizar complejidad de la imagen (submuestreo rápido para detectar pantallas planas o gráficos de otros programas)
        sub = frame[::8, ::8]
        u_colors = len(np.unique(sub.reshape(-1, 3), axis=0))
        cap.release()

        # Clasificación inteligente de cámaras
        if mean_val < 5.0:
            status = "SIN SEÑAL (PANTALLA NEGRA)"
            status_code = "BLACK"
        elif u_colors <= 2:
            status = "PANTALLA PLANA (DESCONECTADA)"
            status_code = "FLAT"
        elif u_colors <= 20:
            status = "LOGO / GRÁFICO VIRTUAL"
            status_code = "LOGO"
        else:
            status = "ACTIVA (EN VIVO)"
            status_code = "ACTIVE"

        # Generar miniatura para el Dashboard (240x135)
        thumb = cv2.resize(frame, (240, 135))

        cams.append({
            "index": i,
            "name": name,
            "opened": True,
            "has_frame": True,
            "status": status,
            "status_code": status_code,
            "res": f"{w}x{h}",
            "mean_brightness": mean_val,
            "unique_colors": u_colors,
            "thumb": thumb
        })

    # Imprimir reporte en terminal
    print(f"{'[ID]':<6} {'Nombre del Dispositivo':<32} {'Resolución':<14} {'Estado':<26}")
    print("-" * 80)
    best_candidate_idx = None

    for c in cams:
        idx_str = f"[{c['index']}]"
        rec_mark = ""
        if c["status_code"] == "ACTIVE":
            if best_candidate_idx is None:
                best_candidate_idx = c["index"]
                rec_mark = " <-- [RECOMENDADA]"
        print(f"{idx_str:<6} {c['name'][:30]:<32} {c['res']:<14} {c['status']:<26}{rec_mark}")

    print("=" * 80 + "\n")

    return cams, best_candidate_idx


def render_dashboard_canvas(cams, selected_idx, hover_idx=None):
    """
    Dibuja la interfaz gráfica del Dashboard de Webcams en un lienzo OpenCV de alta resolución.
    """
    canvas_w = 980
    canvas_h = 660
    canvas = np.full((canvas_h, canvas_w, 3), (28, 28, 34), dtype=np.uint8)

    # 1. Cabecera (Header)
    cv2.rectangle(canvas, (0, 0), (canvas_w, 90), (38, 38, 48), cv2.FILLED)
    cv2.line(canvas, (0, 90), (canvas_w, 90), (0, 200, 220), 2)

    cv2.putText(canvas, "DASHBOARD DE SELECCION DE WEBCAMS", (30, 42),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 235, 235), 2)
    cv2.putText(canvas, "Filtra camaras virtuales inactivas y selecciona la camara real para el Lock Visual",
                (30, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 190, 200), 1)

    # 2. Renderizado de Tarjetas de Cámaras (Grid de 3 columnas x 2 filas)
    card_w = 280
    card_h = 220
    start_x = 40
    start_y = 115
    gap_x = 35
    gap_y = 25

    card_bounds = []

    # Mostrar solo hasta 6 cámaras principales
    visible_cams = cams[:6]

    for idx, c in enumerate(visible_cams):
        row = idx // 3
        col = idx % 3
        cx = start_x + col * (card_w + gap_x)
        cy = start_y + row * (card_h + gap_y)

        card_bounds.append((cx, cy, card_w, card_h, c["index"]))

        is_selected = (c["index"] == selected_idx)
        is_hover = (c["index"] == hover_idx)

        # Color de fondo de la tarjeta
        card_bg = (44, 44, 56) if not is_hover else (54, 54, 68)
        cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h), card_bg, cv2.FILLED)

        # Borde de estado / selección
        if is_selected:
            border_color = (0, 255, 120) # Verde lima neón
            border_thick = 3
        elif is_hover:
            border_color = (255, 200, 0) # Azul/Cian hover
            border_thick = 2
        elif c["status_code"] == "ACTIVE":
            border_color = (70, 140, 70) # Verde apagado
            border_thick = 1
        elif c["status_code"] == "BLACK":
            border_color = (0, 140, 255) # Naranja alerta
            border_thick = 1
        else:
            border_color = (65, 65, 75)  # Gris inactivo
            border_thick = 1

        cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h), border_color, border_thick)

        # Miniatura o placeholder
        thumb_x = cx + 20
        thumb_y = cy + 15
        thumb_w = 240
        thumb_h = 120

        if c["thumb"] is not None:
            # Redimensionar exactamente al área de miniatura
            resized_thumb = cv2.resize(c["thumb"], (thumb_w, thumb_h))
            canvas[thumb_y:thumb_y + thumb_h, thumb_x:thumb_x + thumb_w] = resized_thumb
            cv2.rectangle(canvas, (thumb_x, thumb_y), (thumb_x + thumb_w, thumb_y + thumb_h), (80, 80, 90), 1)
        else:
            # Placeholder oscuro para cámaras sin señal
            cv2.rectangle(canvas, (thumb_x, thumb_y), (thumb_x + thumb_w, thumb_y + thumb_h), (25, 25, 30), cv2.FILLED)
            cv2.rectangle(canvas, (thumb_x, thumb_y), (thumb_x + thumb_w, thumb_y + thumb_h), (50, 50, 60), 1)
            cv2.putText(canvas, "[NO DISPONIBLE]", (thumb_x + 40, thumb_y + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 130), 1)

        # Badge de Estado en la esquina de la miniatura
        if c["status_code"] == "ACTIVE":
            badge_color = (0, 220, 0)
            badge_text = "EN VIVO"
        elif c["status_code"] == "FLAT":
            badge_color = (0, 165, 255)
            badge_text = "SIN CONEXION"
        elif c["status_code"] == "LOGO":
            badge_color = (255, 180, 0)
            badge_text = "VIRTUAL"
        elif c["status_code"] == "BLACK":
            badge_color = (0, 100, 255)
            badge_text = "SIN SEÑAL"
        else:
            badge_color = (100, 100, 100)
            badge_text = "OFFLINE"

        cv2.rectangle(canvas, (thumb_x + 5, thumb_y + 5), (thumb_x + 95, thumb_y + 25), (20, 20, 20), cv2.FILLED)
        cv2.putText(canvas, badge_text, (thumb_x + 10, thumb_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, badge_color, 1)

        # Nombre de la Cámara
        display_name = c["name"]
        if len(display_name) > 24:
            display_name = display_name[:22] + ".."
        cv2.putText(canvas, f"[{c['index']}] {display_name}", (cx + 15, cy + 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Resolución y detalle
        cv2.putText(canvas, f"Res: {c['res']}", (cx + 15, cy + 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 190), 1)

        # Indicador de selección
        if is_selected:
            cv2.rectangle(canvas, (cx + 15, cy + 195), (cx + card_w - 15, cy + 215), (0, 180, 80), cv2.FILLED)
            cv2.putText(canvas, "SELECCIONADA [ENTER]", (cx + 35, cy + 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # 3. Barra Inferior de Controles (Footer)
    cv2.rectangle(canvas, (0, canvas_h - 60), (canvas_w, canvas_h), (35, 35, 45), cv2.FILLED)
    cv2.line(canvas, (0, canvas_h - 60), (canvas_w, canvas_h - 60), (60, 60, 75), 1)

    controls_text = "Teclas [0-5]: Elegir  |  [ENTER/Clic]: Iniciar Lock  |  [R]: Re-escanear  |  [Q/ESC]: Salir"
    cv2.putText(canvas, controls_text, (40, canvas_h - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 230), 1)

    return canvas, card_bounds


def select_webcam_dashboard(default_idx=None):
    """
    Despliega el Dashboard interactivo para que el usuario elija su webcam
    funcional evitando cámaras virtuales de otros programas.
    Retorna el diccionario con la información de la cámara elegida o None si cancela.
    """
    cams, recommended_idx = scan_available_webcams()

    if not cams:
        print("[ERROR] No se detectó ninguna cámara instalada en el sistema.")
        return None

    selected_idx = default_idx if default_idx is not None else (
        recommended_idx if recommended_idx is not None else cams[0]["index"]
    )

    window_name = "Dashboard de Seleccion de Webcams - [Lock Visual]"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    hover_idx = [None]
    confirmed = [False]
    user_quit = [False]

    def mouse_callback(event, x, y, flags, param):
        card_bounds = param["bounds"]
        hover_idx[0] = None

        for (cx, cy, cw, ch, c_idx) in card_bounds:
            if cx <= x <= cx + cw and cy <= y <= cy + ch:
                hover_idx[0] = c_idx
                if event == cv2.EVENT_LBUTTONDOWN:
                    param["selected_idx"][0] = c_idx
                elif event == cv2.EVENT_LBUTTONDBLCLK:
                    param["selected_idx"][0] = c_idx
                    confirmed[0] = True
                break

    selected_holder = [selected_idx]

    while True:
        canvas, bounds = render_dashboard_canvas(cams, selected_holder[0], hover_idx[0])
        cv2.setMouseCallback(window_name, mouse_callback, {"bounds": bounds, "selected_idx": selected_holder})
        cv2.imshow(window_name, canvas)

        if confirmed[0]:
            break

        key = cv2.waitKey(30) & 0xFF

        # Salir con Q o ESC
        if key in [ord('q'), ord('Q'), 27]:
            user_quit[0] = True
            break

        # Confirmar selección con ENTER (13) o ESPACIO (32)
        if key in [13, 32]:
            confirmed[0] = True
            break

        # Teclas numéricas 0-9 para selección rápida
        if ord('0') <= key <= ord('9'):
            pressed_digit = key - ord('0')
            valid_indices = [c["index"] for c in cams]
            if pressed_digit in valid_indices:
                selected_holder[0] = pressed_digit

        # Tecla 'r' para re-escanear
        if key in [ord('r'), ord('R')]:
            print("\n[*] Re-escaneando dispositivos...")
            cams, rec = scan_available_webcams()
            if rec is not None:
                selected_holder[0] = rec

    cv2.destroyWindow(window_name)

    if user_quit[0]:
        print("[*] Selección cancelada por el usuario.")
        return None

    chosen_cam = next((c for c in cams if c["index"] == selected_holder[0]), None)
    if chosen_cam is None:
        chosen_cam = {"index": selected_holder[0], "name": f"Camara {selected_holder[0]}"}

    print(f"\n[+] Cámara seleccionada: [{chosen_cam['index']}] {chosen_cam['name']}")
    return chosen_cam


# =============================================================================
# MÓDULO DE INFERENCIA Y LOCK VISUAL
# =============================================================================

def test_single_image(image_path, model_data, tolerance=0.5):
    """Evalúa una imagen estática y muestra las identidades detectadas."""
    img_resolved = resolve_path(image_path)
    if not os.path.exists(img_resolved):
        print(f"[ERROR] No existe la imagen: {img_resolved}")
        return

    image = cv2.imread(img_resolved)
    if image is None:
        print(f"[ERROR] No se pudo leer la imagen: {img_resolved}")
        return

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    boxes = face_recognition.face_locations(rgb, model="hog")
    if not boxes and h > 20 and w > 20:
        boxes = [(0, w, h, 0)]
    encodings = face_recognition.face_encodings(rgb, boxes)

    print(f"[*] Rostros detectados en la imagen: {len(boxes)}")
    for (top, right, bottom, left), encoding in zip(boxes, encodings):
        distances = face_recognition.face_distance(model_data["encodings"], encoding)
        min_distance = np.min(distances) if len(distances) > 0 else 1.0
        best_match_idx = np.argmin(distances) if len(distances) > 0 else -1

        if min_distance <= tolerance and best_match_idx != -1:
            name = model_data["names"][best_match_idx]
            confidence = (1.0 - min_distance) * 100
            print(f" -> [ACCESO AUTORIZADO] Sujeto: {name} | Similitud: {confidence:.1f}% (distancia: {min_distance:.3f})")
        else:
            confidence = (1.0 - min_distance) * 100 if min_distance <= 1.0 else 0
            print(f" -> [ACCESO DENEGADO / LOCK ACTIVO] Rostro no reconocido | Distancia mínima: {min_distance:.3f}")


def run_webcam_lock(model_data, tolerance=0.5, camera_info=None):
    """
    Inicia el flujo de video en vivo para simular el Lock Visual.
    Soporta cambio de cámara en caliente presionando la tecla 'c'.
    """
    current_camera = camera_info

    # Si no se pasó una cámara específica, abrir el Dashboard de selección
    if current_camera is None:
        current_camera = select_webcam_dashboard()
        if current_camera is None:
            return

    while True:
        cam_idx = current_camera["index"]
        cam_name = current_camera.get("name", f"Camara {cam_idx}")

        print("=" * 70)
        print(f"   INICIANDO LOCK VISUAL CON CÁMARA [{cam_idx}]: {cam_name}")
        print("   -> Tecla 'c': Cambiar de webcam (Abrir Dashboard)")
        print("   -> Tecla 'q': Salir del sistema")
        print("=" * 70)

        # Intentar DirectShow primero en Windows
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_idx)

        if not cap.isOpened():
            print(f"[ERROR] No se pudo abrir la cámara [{cam_idx}]: {cam_name}.")
            print("Abriendo dashboard para seleccionar otra...")
            current_camera = select_webcam_dashboard(default_idx=cam_idx)
            if current_camera is None:
                break
            continue

        # Configurar resolución recomendada
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        process_current_frame = True
        switch_requested = False
        window_title = f"Lock Visual Biometrico - [{cam_name}]"

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"[!] No se pudo recibir fotograma de la cámara [{cam_idx}].")
                break

            # Procesar 1 de cada 2 fotogramas para mantener 30+ FPS en CPU
            if process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                confidences = []

                for face_encoding in face_encodings:
                    distances = face_recognition.face_distance(model_data["encodings"], face_encoding)
                    min_distance = np.min(distances) if len(distances) > 0 else 1.0
                    best_match_idx = np.argmin(distances) if len(distances) > 0 else -1

                    if min_distance <= tolerance and best_match_idx != -1:
                        name = model_data["names"][best_match_idx]
                        conf = (1.0 - min_distance) * 100
                    else:
                        name = "BLOQUEADO"
                        conf = (1.0 - min_distance) * 100 if min_distance <= 1.0 else 0

                    face_names.append(name)
                    confidences.append(conf)

            process_current_frame = not process_current_frame

            # Dibujar rectángulos y etiquetas sobre los rostros
            for (top, right, bottom, left), name, conf in zip(face_locations, face_names, confidences):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                if name == "BLOQUEADO":
                    color = (0, 0, 255) # Rojo
                    status_text = f"LOCK: NO AUTORIZADO ({conf:.1f}%)"
                else:
                    color = (0, 255, 0) # Verde
                    status_text = f"DESBLOQUEADO: {name} ({conf:.1f}%)"

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 32), (right, bottom), color, cv2.FILLED)
                cv2.putText(
                    frame,
                    status_text,
                    (left + 6, bottom - 8),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.55,
                    (255, 255, 255),
                    1
                )

            # HUD Superior: Estado del Lock Visual
            if face_names:
                any_authorized = any(n != "BLOQUEADO" for n in face_names)
                hud_color = (0, 220, 0) if any_authorized else (0, 0, 255)
                hud_text = "ESTADO: ACCESO CONCEDIDO" if any_authorized else "ESTADO: BLOQUEO ACTIVO"
            else:
                hud_color = (0, 165, 255) # Naranja
                hud_text = "ESTADO: ESPERANDO ROSTRO..."

            # Barra superior semitransparente
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 45), (20, 20, 25), cv2.FILLED)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, hud_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, hud_color, 2)

            # Barra inferior con información de cámara y controles
            overlay_bot = frame.copy()
            fh, fw = frame.shape[:2]
            cv2.rectangle(overlay_bot, (0, fh - 35), (fw, fh), (20, 20, 25), cv2.FILLED)
            cv2.addWeighted(overlay_bot, 0.7, frame, 0.3, 0, frame)

            footer_info = f"Camara: [{cam_idx}] {cam_name[:24]} | 'c': Cambiar webcam | 'q': Salir"
            cv2.putText(frame, footer_info, (15, fh - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 235, 235), 1)

            cv2.imshow(window_title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break

            # Tecla 'c' para abrir el Dashboard y cambiar de cámara en caliente
            if key in [ord('c'), ord('C')]:
                switch_requested = True
                break

        cap.release()
        cv2.destroyWindow(window_title)

        if switch_requested:
            print("\n[*] Abriendo Dashboard de Selección de Webcams...")
            new_cam = select_webcam_dashboard(default_idx=cam_idx)
            if new_cam is not None:
                current_camera = new_cam
            # Si cancela, mantiene la cámara actual en el siguiente ciclo
        else:
            # Salida solicitada por el usuario ('q')
            break

    cv2.destroyAllWindows()
    print("[*] Sistema Lock Visual cerrado correctamente.")


# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sistema de Inferencia y Lock Visual con Dashboard de Webcams."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="modelo_biometrico.pkl",
        help="Ruta del archivo modelo_biometrico.pkl"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Ruta de una imagen estática para prueba directa sin cámara"
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Iniciar en modo cámara web interactiva con Dashboard"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Índice numérico de la cámara si ya se conoce (ej: 0, 4) para omitir el selector inicial"
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Escanear y mostrar la tabla de webcams en la terminal sin iniciar reconocimiento"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="Umbral de tolerancia euclidiana (menor = más estricto, default: 0.5)"
    )

    args = parser.parse_args()

    # Si se pide solo listar cámaras
    if args.list_cameras:
        scan_available_webcams()
        sys.exit(0)

    model = load_biometric_model(args.model)

    if args.image:
        test_single_image(args.image, model, args.tolerance)
    elif args.webcam or args.camera is not None:
        cam_info = None
        if args.camera is not None:
            cam_info = {"index": args.camera, "name": f"Camara {args.camera}"}
        run_webcam_lock(model, args.tolerance, camera_info=cam_info)
    else:
        # Por defecto, iniciar el Dashboard de selección de webcams
        print("[*] No se especificaron banderas. Iniciando Dashboard de Webcams...")
        run_webcam_lock(model, args.tolerance, camera_info=None)
