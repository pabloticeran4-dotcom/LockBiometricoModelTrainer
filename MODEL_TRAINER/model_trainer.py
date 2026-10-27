"""
=============================================================================
MÓDULO DE ENTRENAMIENTO DE MODELO BIOMÉTRICO (FACE ENCODINGS)
Proyecto: Lock Visual / Control de Acceso Facial
Autor: Pablo Ticeran
=============================================================================
Este script procesa las identidades en el directorio del dataset, extrae los
descriptores biométricos faciales (embeddings de 128 dimensiones) usando dlib
y face_recognition, y genera un archivo serializado (.pkl) listo para ser
utilizado por el sistema de autenticación y bloqueo visual.
=============================================================================
"""

import os
import sys
import time
import pickle
import cv2
import face_recognition

def resolve_path(target_path):
    """
    Resuelve una ruta relativa buscando primero en el directorio de trabajo actual
    y luego relativo a la raíz del proyecto (directorio padre de MODEL_TRAINER).
    """
    if os.path.isabs(target_path) and os.path.exists(target_path):
        return target_path
        
    # Verificar en cwd
    if os.path.exists(target_path):
        return os.path.abspath(target_path)
        
    # Verificar respecto a la raíz del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    root_relative = os.path.join(project_root, target_path)
    if os.path.exists(root_relative):
        return root_relative
        
    # Si es ruta de salida que aún no existe, preferir la raíz del proyecto si estamos en MODEL_TRAINER
    if os.path.basename(os.getcwd()).lower() == "model_trainer":
        return root_relative
        
    return os.path.abspath(target_path)


def train_multi_face_model(base_dir="dataset_proyecto", model_save_path="modelo_biometrico.pkl", detection_model="hog"):
    """
    Entrena el modelo extrayendo los 128 descriptores por rostro en el dataset.
    
    Args:
        base_dir: Directorio con subcarpetas para cada persona (ej: dataset_proyecto/Pablo).
        model_save_path: Ruta del archivo .pkl donde se guardará el modelo entrenado.
        detection_model: Modelo de detección ('hog' para CPU rápido, 'cnn' si hay GPU CUDA).
    """
    base_dir_resolved = resolve_path(base_dir)
    save_path_resolved = resolve_path(model_save_path)
    
    print("=" * 65)
    print("    SISTEMA DE ENTRENAMIENTO BIOMÉTRICO - LOCK VISUAL")
    print("=" * 65)
    print(f"[*] Directorio de dataset: {base_dir_resolved}")
    print(f"[*] Ruta de salida modelo: {save_path_resolved}")
    print(f"[*] Algoritmo de detección: {detection_model.upper()}")
    print("-" * 65)

    if not os.path.exists(base_dir_resolved) or not os.path.isdir(base_dir_resolved):
        print(f"[ERROR] No se encontró el directorio de dataset: '{base_dir_resolved}'")
        print("Asegúrate de que la carpeta existe y contiene las identidades.")
        return False

    known_encodings = []
    known_names = []
    
    start_time = time.time()
    total_images_processed = 0
    total_faces_detected = 0
    
    subdirs = [d for d in os.listdir(base_dir_resolved) if os.path.isdir(os.path.join(base_dir_resolved, d))]
    
    if not subdirs:
        print(f"[ADVERTENCIA] No se encontraron subcarpetas de personas en: {base_dir_resolved}")
        return False

    print(f"[INFO] Identidades encontradas en dataset ({len(subdirs)}): {subdirs}\n")

    for person_name in subdirs:
        person_dir = os.path.join(base_dir_resolved, person_name)
        image_files = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        
        print(f">> Procesando identidad: '{person_name}' ({len(image_files)} imágenes encontradas)")
        person_encodings_count = 0

        for idx, filename in enumerate(image_files, 1):
            image_path = os.path.join(person_dir, filename)
            total_images_processed += 1
            
            # Cargar imagen con OpenCV
            image = cv2.imread(image_path)
            if image is None:
                print(f"   [!] Error al leer: {filename} (saltando)")
                continue
                
            # Convertir de BGR (OpenCV) a RGB (face_recognition / dlib)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            # 1. Intentar localizar el rostro con el modelo HOG / CNN
            boxes = face_recognition.face_locations(rgb, model=detection_model)
            
            # 2. Respaldo para imágenes previamente recortadas por Haar Cascades
            # Si HOG no detecta caja porque la imagen ya está recortada al ras,
            # usamos las dimensiones completas de la imagen como caja.
            if not boxes and h > 20 and w > 20:
                boxes = [(0, w, h, 0)]
                
            encodings = face_recognition.face_encodings(rgb, boxes)

            if encodings:
                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(person_name)
                    person_encodings_count += 1
                    total_faces_detected += 1
            else:
                print(f"   [-] No se pudo extraer descriptor para: {filename}")

            # Mostrar progreso cada 25 imágenes o al final
            if idx % 25 == 0 or idx == len(image_files):
                print(f"   Progreso '{person_name}': {idx}/{len(image_files)} procesadas...")

        print(f">> Finalizada identidad '{person_name}': {person_encodings_count} descriptores extraídos.\n")

    if not known_encodings:
        print("[ERROR] No se pudo extraer ningún descriptor facial del dataset.")
        return False

    # 3. Guardar el modelo biométrico compilado en formato pickle
    print("-" * 65)
    print(f"[INFO] Guardando modelo en: {save_path_resolved}")
    data = {
        "encodings": known_encodings,
        "names": known_names,
        "classes": list(sorted(set(known_names))),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Asegurar que el directorio de salida exista
    output_dir = os.path.dirname(save_path_resolved)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    with open(save_path_resolved, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_kb = os.path.getsize(save_path_resolved) / 1024
    elapsed = time.time() - start_time
    
    print("=" * 65)
    print("           ¡ENTRENAMIENTO BIOMÉTRICO COMPLETADO CON ÉXITO!")
    print("=" * 65)
    print(f"[*] Tiempo transcurrido   : {elapsed:.2f} segundos")
    print(f"[*] Imágenes procesadas   : {total_images_processed}")
    print(f"[*] Descriptores creados  : {total_faces_detected}")
    print(f"[*] Identidades activas   : {list(set(known_names))}")
    print(f"[*] Tamaño del modelo     : {file_size_kb:.2f} KB")
    print("=" * 65)
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Entrena y compila el modelo biométrico de reconocimiento facial para el Lock Visual."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset_proyecto",
        help="Carpeta que contiene las subcarpetas con los nombres de las personas (default: dataset_proyecto)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="modelo_biometrico.pkl",
        help="Archivo donde se guardará el modelo compilado (default: modelo_biometrico.pkl)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["hog", "cnn"],
        default="hog",
        help="Algoritmo de detección: 'hog' para CPU rápida, 'cnn' para GPU dlib CUDA (default: hog)"
    )

    args = parser.parse_args()
    train_multi_face_model(
        base_dir=args.dataset,
        model_save_path=args.output,
        detection_model=args.model
    )
