"""
=============================================================================
EXTRACTOR DE ROSTROS DESDE VIDEO (DATASET BUILDER)
Proyecto: Lock Visual / Control de Acceso Facial
Autor: Pablo Ticeran
=============================================================================
Este script extrae fotogramas con rostros detectados desde videos (.MTS, .mp4, etc.)
utilizando los clasificadores Haar Cascade de OpenCV. Posee control para evitar
sobreescritura de imágenes existentes incrementando el contador numérico.
=============================================================================
"""

import cv2
import os

def resolve_video_path(video_path):
    """Verifica si la ruta del video existe en CWD o en la carpeta del script."""
    if os.path.exists(video_path):
        return video_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, video_path)
    if os.path.exists(candidate):
        return candidate
    return video_path


def extract_faces_from_video(video_path, output_dir, max_faces=50, frames_to_skip=15):
    """
    Extrae rostros de un video y los guarda como imágenes .jpg independientes.
    
    Args:
        video_path: Ruta del video de entrada (.MTS, .mp4, .avi, etc.)
        output_dir: Directorio destino para los recortes de rostro
        max_faces: Límite máximo de rostros a extraer por video
        frames_to_skip: Intervalo de fotogramas a saltar para capturar variación
    """
    video_path = resolve_video_path(video_path)
    
    if not os.path.exists(video_path):
        print(f"[ERROR] No se encuentra el video en: {video_path}")
        return

    # Crear directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # --- LÓGICA: Buscar el último número para no sobreescribir ---
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("video_rostro_") and f.endswith(".jpg")]
    start_index = 0
    for f in existing_files:
        try:
            # Extraer el número del nombre (ej: "video_rostro_15.jpg" -> 15)
            num = int(f.split('_')[-1].split('.')[0])
            if num > start_index:
                start_index = num
        except ValueError:
            pass
            
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir el archivo de video: {video_path}")
        return
    
    count_saved = 0
    frame_count = 0
    
    print("-" * 60)
    print(f"Procesando video: {video_path}")
    print(f"Las imágenes continuarán desde el número: {start_index + 1}")
    
    while cap.isOpened() and count_saved < max_faces:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        if frame_count % frames_to_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                # Recortar el rostro
                face_img = frame[y:y+h, x:x+w]
                
                # Guardar imagen sumando el índice inicial
                filename = os.path.join(output_dir, f"video_rostro_{start_index + count_saved + 1}.jpg")
                cv2.imwrite(filename, face_img)
                count_saved += 1
                
                print(f"Guardada imagen: {filename}")
                break 
                
    cap.release()
    print(f"Extracción completada. {count_saved} rostros nuevos guardados en {output_dir}")
    print("-" * 60)


if __name__ == "__main__":
    # Procesa videos por lotes si se ejecuta directamente:
    videos = ["00147.MTS", "00148.MTS", "00149.MTS", "00150.MTS", "00151.MTS", "00152.MTS"]
    output_folder = "dataset_mi_rostro_video"
    
    for v in videos:
        resolved = resolve_video_path(v)
        if os.path.exists(resolved):
            extract_faces_from_video(resolved, output_folder, max_faces=50, frames_to_skip=15)
        else:
            print(f"[SKIP] Video no encontrado: {v}")

if __name__ == "__main__":
    # Procesa videos por lotes si se ejecuta directamente:
    videos = ["00147.MTS", "00148.MTS", "00149.MTS", "00150.MTS", "00151.MTS", "00152.MTS"]
    output_folder = "dataset_mi_rostro_video"
    
    for v in videos:
        resolved = resolve_video_path(v)
        if os.path.exists(resolved):
            extract_faces_from_video(resolved, output_folder, max_faces=50, frames_to_skip=15)
        else:
            print(f"[SKIP] Video no encontrado: {v}")

