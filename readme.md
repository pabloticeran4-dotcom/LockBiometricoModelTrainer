# Sistema de Lock Visual Biométrico (ModelAeaFotoPAB)

**Autor:** Pablo Ticeran  
**Tecnologías:** Python 3.14 | OpenCV | dlib | face_recognition | NumPy | pygrabber  
**Estado:** Fase 2 Completada (Modelo Entrenado) & Fase 3 Operativa (Dashboard de Webcams + Lock Visual)  

---

## 📌 Descripción del Proyecto

**ModelAeaFotoPAB** es un sistema biométrico local de **Lock Visual** (bloqueo y desbloqueo de acceso mediante visión computacional y reconocimiento facial). El proyecto está diseñado para funcionar 100% de manera local aprovechando la capacidad de hardware de la máquina (CPU/GPU), sin enviar flujos de video ni datos biométricos a la nube.

### Objetivos del Proyecto:
- [x] **Extracción de Rostros:** Scripts para procesar secuencias de video y extraer recortes limpios de rostros con control de índice incremental.
- [x] **Entrenamiento Local de Embeddings:** Algoritmos para vectorizar descriptores faciales (128 dimensiones) y serializar un modelo biométrico binario optimizado (`.pkl`).
- [x] **Dashboard de Selección de Webcams:** Interfaz gráfica interactiva para previsualizar, diagnosticar y seleccionar cámaras reales, filtrando dispositivos virtuales inactivos (OBS, DroidCam sin conectar, NVIDIA Broadcast, etc.).
- [x] **Prueba y Validación en Tiempo Real (Lock Visual):** Inferencia en tiempo real mediante cámara web con HUD visual dinámico y soporte para cambio de cámara en caliente.
- [ ] **Bloqueo a Nivel de Sistema Operativo:** Integración directa con las APIs del sistema para bloquear la estación de trabajo ante usuarios desconocidos o ausencia.

---

## 🚀 Estado Actual del Proyecto (Avance)

El proyecto se encuentra en la **Fase 2 finalizada y Fase 3 en desarrollo activo**:

1. **Dataset Extraído:** 112 imágenes de rostros procesadas y filtradas bajo la identidad `Pablo`.
2. **Modelo Biométrico Compilado:** Se ha generado con éxito el archivo `modelo_biometrico.pkl` conteniendo los 112 descriptores normalizados.
3. **Dashboard de Webcams Integrado:** Interfaz interactiva de OpenCV que analiza la señal de video de cada cámara instalada en Windows, muestra miniaturas en vivo y clasifica su estado (`EN VIVO`, `SIN CONEXIÓN`, `VIRTUAL`, `SIN SEÑAL`, `OFFLINE`).
4. **Lock Visual en Tiempo Real:** Módulo `LOCK_VISUAL/visual_lock.py` con interfaz en vivo por webcam, cálculo de distancia euclidiana, cálculo de porcentaje de similitud y recuadros dinámicos (Verde para usuario autorizado, Rojo para bloqueo).
5. **Problema de Incompatibilidad Resuelto:** Se corrigió el fallo con `face_recognition_models` / `pkg_resources` en Python 3.14 mediante la estabilización de `setuptools<70`.

Para mayor detalle técnico, consulta [Estado del Proyecto](file:///docs/ESTADO_DEL_PROYECTO.md).

---

## 🎥 Dashboard de Selección de Webcams

Si tienes instalados programas como **OBS Studio**, **DroidCam**, **NVIDIA Broadcast** o **Eyeware**, Windows registra múltiples cámaras virtuales que con frecuencia están inactivas o muestran pantallas negras/planas.

El archivo `LOCK_VISUAL/visual_lock.py` incluye un Dashboard que:
- Detecta los nombres reales de los dispositivos DirectShow de Windows.
- Inspecciona los fotogramas y diagnostica si la cámara tiene señal real o es un gráfico estático.
- Muestra una cuadrícula de tarjetas con miniatura de previsualización para cada cámara.
- Permite seleccionar la cámara deseada haciendo clic con el ratón o presionando las teclas numéricas `[0-5]`.
- Permite cambiar de cámara en cualquier momento durante la sesión presionando la tecla **`c`**.

---

## 📂 Estructura del Repositorio

```text
ModelAeaFotoPAB/
├── .venv/                      # Entorno virtual de Python configurado
├── DATABASE_EXTRACTOR/         # Extracción de dataset desde videos
│   ├── face_extractor_vid.py   # Script de recorte facial con Haar Cascades
│   ├── 00147.MTS ... 00152.MTS # Videos fuente de muestra
│   └── video_rostro_*.jpg      # Rostros extraídos
├── dataset_proyecto/           # Dataset organizado por identidades
│   └── Pablo/                  # 112 imágenes de muestra de Pablo
├── MODEL_TRAINER/              # Entrenamiento y vectorización
│   └── model_trainer.py        # Compilador de embeddings biométricos (dlib HOG/CNN)
├── LOCK_VISUAL/                # Inferencia y sistema de bloqueo visual
│   └── visual_lock.py          # Dashboard de webcams y reconocimiento en vivo
├── docs/                       # Documentación técnica extendida
│   ├── ESTADO_DEL_PROYECTO.md
│   ├── SOLUCION_ERROR_FACE_RECOGNITION.md
│   └── GUIA_DE_USO_Y_COMANDOS.md
├── modelo_biometrico.pkl       # Modelo compilado listo para inferencia
├── requirements.txt            # Dependencias del proyecto
└── readme.md                   # Documentación principal
```

---

## ⚙️ Instalación y Requisitos

### Requisitos Previos
- Python 3.10 a 3.14 instalado en Windows.
- Cámara web (física o mediante DroidCam/celular conectado).

### Configuración del Entorno
1. Clonar o abrir la carpeta del proyecto.
2. Activar el entorno virtual:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
3. Instalar las dependencias fijadas en `requirements.txt`:
   ```powershell
   pip install -r requirements.txt
   ```

> [!NOTE]
> `face_recognition_models` requiere `setuptools < 70` debido a la dependencia interna de `pkg_resources`. Esto ya se encuentra configurado en `requirements.txt`. Para ver la explicación técnica del problema y su solución, consulta [Solución de Error face_recognition_models](file:///docs/SOLUCION_ERROR_FACE_RECOGNITION.md).

---

## 🛠️ Guía Rápida de Ejecución

### 1. Entrenar el Modelo con el Dataset
```powershell
python MODEL_TRAINER/model_trainer.py
```
*Procesará todas las subcarpetas dentro de `dataset_proyecto/` y creará `modelo_biometrico.pkl`.*

### 2. Probar el Modelo sobre una Imagen Estática
```powershell
python LOCK_VISUAL/visual_lock.py --image "dataset_proyecto/Pablo/video_rostro_1.jpg"
```

### 3. Iniciar el Lock Visual con el Dashboard de Webcams
```powershell
python LOCK_VISUAL/visual_lock.py --webcam
```
O simplemente:
```powershell
python LOCK_VISUAL/visual_lock.py
```

### 4. Opciones de Línea de Comandos Disponibles
```powershell
# Solo listar cámaras detectadas y su estado en la consola:
python LOCK_VISUAL/visual_lock.py --list-cameras

# Iniciar directamente con un índice de cámara conocido (ej: cámara 4):
python LOCK_VISUAL/visual_lock.py --camera 4

# Ajustar tolerancia euclidiana (menor = más estricto, default: 0.5):
python LOCK_VISUAL/visual_lock.py --tolerance 0.45
```

---

## 🎮 Controles Durante la Ejecución

| Pantalla | Tecla / Acción | Función |
| :--- | :--- | :--- |
| **Dashboard** | `0` a `5` | Seleccionar la cámara correspondiente a ese número. |
| **Dashboard** | `Clic izquierdo` | Seleccionar tarjeta de cámara. |
| **Dashboard** | `Doble clic` / `ENTER` | Confirmar e iniciar Lock Visual con la cámara seleccionada. |
| **Dashboard** | `R` | Re-escanear dispositivos de video. |
| **Dashboard** | `Q` / `ESC` | Cancelar y salir. |
| **Lock Visual** | `C` | **Cambiar de cámara en caliente** (reabre el Dashboard sin cerrar la app). |
| **Lock Visual** | `Q` / `ESC` | Salir del sistema Lock Visual. |

Para conocer más detalles, consulta la [Guía de Uso y Comandos](file:///docs/GUIA_DE_USO_Y_COMANDOS.md).

---

## 🔬 Métodos y Algoritmos Utilizados

- **Detección Previa:** OpenCV Haar Cascade Classifiers (`haarcascade_frontalface_default.xml`).
- **Alineación y Landmarks:** dlib 68-Point Face Landmark Predictor.
- **Extracción de Características:** Red Neuronal Convolucional Profunda ResNet-34 entrenada en el espacio euclidiano ($\mathbb{R}^{128}$).
- **Clasificación y Validación:** Distancia Euclidiana L2 con umbral calibrado a 0.50.
- **Enumeración DirectShow:** `pygrabber` y `cv2.CAP_DSHOW` para detección nativa de hardware en Windows.
