# Guía de Uso y Comandos del Proyecto

**Proyecto:** Lock Visual Biométrico (ModelAeaFotoPAB)  
**Autor:** Pablo Ticeran  

---

## 1. Activación del Entorno Virtual

Antes de ejecutar cualquier script, abre una terminal de PowerShell en la carpeta raíz del proyecto y activa el entorno:

```powershell
.\.venv\Scripts\Activate.ps1
```

O si utilizas el Command Prompt (cmd):
```cmd
.\.venv\Scripts\activate.bat
```

---

## 2. Flujo Completo de Trabajo

### Paso 1: Extracción de Rostros desde Videos
Si deseas agregar más videos o capturar rostros adicionales:
1. Coloca los archivos de video (`.MTS`, `.mp4`) dentro de `DATABASE_EXTRACTOR/`.
2. Ejecuta el extractor:
```powershell
python DATABASE_EXTRACTOR/face_extractor_vid.py
```
3. Los rostros recortados se guardarán con numeración correlativa (`video_rostro_1.jpg`, `video_rostro_2.jpg`, etc.) evitando sobreescritura.
4. Mueve o copia las imágenes seleccionadas a la carpeta correspondiente dentro de `dataset_proyecto/<NombrePersona>/`.

---

### Paso 2: Entrenamiento del Modelo Biométrico
Para compilar todas las identidades y rostros en el modelo `modelo_biometrico.pkl`:

**Ejecución básica:**
```powershell
python MODEL_TRAINER/model_trainer.py
```

**Ejecución con parámetros personalizados:**
```powershell
# Especificando ruta de dataset y nombre de salida
python MODEL_TRAINER/model_trainer.py --dataset dataset_proyecto --output modelo_biometrico.pkl

# Usando detección CNN en lugar de HOG (requiere GPU con CUDA compilado en dlib)
python MODEL_TRAINER/model_trainer.py --model cnn
```

El script mostrará el progreso en tiempo real y confirmará la cantidad de imágenes procesadas y el tamaño final del archivo `.pkl`.

---

### Paso 3: Probar el Modelo y el Sistema Lock Visual

#### A) Dashboard de Selección de Webcams (Recomendado)
Dado que muchos programas (OBS, DroidCam, NVIDIA Broadcast, Eyeware) crean cámaras virtuales inactivas o en negro, el sistema incorpora un Dashboard visual interactivo:

```powershell
python LOCK_VISUAL/visual_lock.py --webcam
```
*(o simplemente `python LOCK_VISUAL/visual_lock.py`)*

**Cómo funciona el Dashboard:**
- Analiza cada dispositivo y genera una miniatura con su estado en tiempo real.
- **`EN VIVO` (Verde):** Cámara con señal de video activa y variación de imagen.
- **`SIN CONEXIÓN` (Naranja):** Cámara de programa virtual pero sin feed conectado (ej: DroidCam esperando conexión del móvil).
- **`VIRTUAL` (Azul/Amarillo):** Muestra un logotipo o pantalla de espera estática (ej: Eyeware o NVIDIA Broadcast).
- **`SIN SEÑAL` (Rojo):** Emite pantalla totalmente negra.
- **`OFFLINE` (Gris):** Dispositivo ocupado o programa cerrado.

**Controles del Dashboard:**
- **Seleccionar con Ratón:** Haz un clic sobre la tarjeta de la cámara deseada.
- **Iniciar Inmediatamente:** Haz doble clic sobre la tarjeta o presiona **`ENTER`**.
- **Teclas Numéricas:** Presiona `0`, `1`, `2`, `3`, `4`, `5` para elegir la cámara por su número.
- **Re-escanear:** Presiona **`R`** si acabas de conectar tu celular con DroidCam o enchufar una webcam USB.
- **Salir:** Presiona **`Q`** o **`ESC`**.

#### B) Cambio de Cámara en Caliente durante el Lock Visual
Si ya estás dentro de la ventana de reconocimiento facial y quieres cambiar de cámara:
- Presiona la tecla **`C`**.
- La transmisión actual se detendrá y se reabrirá el Dashboard para que elijas otra cámara sin cerrar la aplicación.

#### C) Listado Rápido por Consola (Sin Interfaz)
Para ver un diagnóstico rápido de las webcams en texto:
```powershell
python LOCK_VISUAL/visual_lock.py --list-cameras
```

#### D) Iniciar Directamente con una Cámara Específica
Si ya sabes que tu cámara es la número 4 o la 0, puedes saltarte el selector inicial:
```powershell
python LOCK_VISUAL/visual_lock.py --camera 4
```

#### E) Prueba con Imagen Estática
```powershell
python LOCK_VISUAL/visual_lock.py --image "dataset_proyecto/Pablo/video_rostro_1.jpg"
```

#### F) Ajuste del Umbral de Tolerancia
Puedes controlar cuán estricto es el sistema mediante el parámetro `--tolerance` (por defecto `0.5`):
```powershell
# Mayor seguridad (más estricto, menor tolerancia a cambios)
python LOCK_VISUAL/visual_lock.py --webcam --tolerance 0.45

# Mayor permisividad (útil si hay baja iluminación)
python LOCK_VISUAL/visual_lock.py --webcam --tolerance 0.55
```

---

## 3. Estructura de Directorios para Nuevas Identidades

Para registrar múltiples usuarios en el Lock Visual, simplemente organiza las carpetas dentro de `dataset_proyecto` de la siguiente manera:

```text
dataset_proyecto/
├── Pablo/
│   ├── rostro_1.jpg
│   ├── rostro_2.jpg
│   └── ...
├── Usuario_Invitado/
│   ├── rostro_1.jpg
│   └── ...
└── Administrador/
    ├── rostro_1.jpg
    └── ...
```

Al reejecutar `python MODEL_TRAINER/model_trainer.py`, el sistema detectará automáticamente cada carpeta como una identidad distinta y las compilará en `modelo_biometrico.pkl`.
