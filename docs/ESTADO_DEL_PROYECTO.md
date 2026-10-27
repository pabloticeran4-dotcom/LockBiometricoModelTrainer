# Estado del Proyecto: Lock Visual Biométrico (ModelAeaFotoPAB)

**Autor:** Pablo Ticeran  
**Fecha de Actualización:** 27 de Octubre de 2026  
**Versión del Sistema:** v0.3.0 (Fase 2 Completada + Dashboard de Webcams y Prototipo de Lock Visual Operativo)

---

## 1. Resumen Ejecutivo y Objetivo

El proyecto **Lock Visual** consiste en un sistema inteligente de seguridad y control de acceso biométrico basado en reconocimiento facial. El objetivo es permitir el bloqueo o desbloqueo visual local de una estación de trabajo o sesión aprovechando el hardware local (procesamiento en CPU/GPU mediante OpenCV y redes neuronales profundas con dlib), sin depender de servicios en la nube.

---

## 2. Mapa de Fases del Proyecto

| Fase | Descripción | Estado | Componente Principal |
| :--- | :--- | :---: | :--- |
| **Fase 1: Extracción de Dataset** | Procesamiento de video (.MTS / .mp4), recorte automático de rostros con Haar Cascades y numeración secuencial. | **COMPLETADO** | `DATABASE_EXTRACTOR/face_extractor_vid.py` |
| **Fase 2: Vectorización y Entrenamiento** | Extracción de embeddings faciales de 128 dimensiones (HOG / ResNet) y compilación a archivo serializado `.pkl`. | **COMPLETADO** | `MODEL_TRAINER/model_trainer.py` |
| **Fase 3: Dashboard de Webcams e Inferencia en Tiempo Real** | Selector interactivo de cámaras con filtrado de programas virtuales inactivos + Reconocimiento en vivo con HUD de estado. | **COMPLETADO / OPERATIVO** | `LOCK_VISUAL/visual_lock.py` |
| **Fase 4: Integración de Bloqueo a Nivel de Sistema** | Bloqueo automático de pantalla de Windows (`ctypes.windll.user32.LockWorkStation()`) al detectar usuario desconocido o ausencia. | **PLANEADO** | `LOCK_VISUAL/system_lock_service.py` |
| **Fase 5: Anti-Spoofing (Prueba de Vida)** | Detección de parpadeo o movimiento tridimensional para evitar engaños mediante fotografías en papel o pantallas de móvil. | **PLANEADO** | Módulo de Liveness Detection |

---

## 3. Detalle de la Etapa Actual (Fase 2 y Fase 3)

### Logros Alcanzados en esta Etapa:
1. **Resolución Crítica de Compatibilidad:**
   - Se diagnosticó y corrigió el problema de incompatibilidad entre `face_recognition_models` y versiones modernas de `setuptools` en Python 3.14.
   - El entorno virtual ahora carga e inicializa correctamente todos los pesos de la red neuronal y modelos de landmark (68 puntos) de dlib.

2. **Entrenamiento y Generación del Modelo Biométrico:**
   - Script `MODEL_TRAINER/model_trainer.py` optimizado y ejecutado satisfactoriamente.
   - Se procesó el dataset de la identidad **Pablo** con **112 imágenes** procesadas con una tasa de éxito del 100%.
   - Se generó el archivo compilado `modelo_biometrico.pkl` (115.47 KB) que almacena los 112 vectores biométricos de 128 flotantes junto con sus etiquetas e índices.

3. **Dashboard de Selección de Webcams:**
   - Implementado en `LOCK_VISUAL/visual_lock.py`.
   - Detecta y clasifica dispositivos DirectShow en Windows (OBS, DroidCam, NVIDIA Broadcast, Eyeware, Webcams USB).
   - Genera miniaturas en tiempo real y discrimina automáticamente entre cámaras con señal en vivo, pantallas negras, gráficos estáticos y programas desconectados.
   - Permite seleccionar la cámara por ratón (clic / doble clic) o teclas numéricas (`0` a `5`).
   - Permite cambiar de cámara en caliente presionando la tecla **`C`** durante la ejecución.

4. **Verificación y Prueba de Inferencia:**
   - Reconocimiento facial en tiempo real a 30+ FPS en CPU con reducción cuadrática de escala.
   - HUD dinámico: Recuadro verde con porcentaje de similitud para usuarios autorizados (`DESBLOQUEADO: Pablo`) y recuadro rojo para accesos no autorizados (`LOCK: NO AUTORIZADO`).

---

## 4. Métricas Técnicas Actuales

- **Tiempo de Entrenamiento del Modelo:** 26.87 segundos (112 imágenes en CPU).
- **Rendimiento de Extracción:** ~0.24 segundos por imagen.
- **Espacio de Características:** Vector de 128 dimensiones normalizado por rostro.
- **Métrica de Distancia:** Distancia Euclidiana en $\mathbb{R}^{128}$.
- **Umbral de Tolerancia Recomendado:**
  - $\text{Distancia} \le 0.50$: Coincidencia segura (Acceso Concedido).
  - $\text{Distancia} > 0.50$: No reconocido / Bloqueo activo.

---

## 5. Próximos Pasos (Siguiente Avance)

1. **Adición de Nuevas Identidades:**
   - Agregar carpetas adicionales en `dataset_proyecto/` para validar el rechazo de personas no autorizadas frente a usuarios permitidos.
2. **Mecanismo de Desconexión / Bloqueo de Sesión:**
   - Integrar llamada a la API Win32 `LockWorkStation` cuando un usuario no autorizado se presente frente a la cámara o transcurra un tiempo de gracia sin detección.
3. **Módulo de Prueba de Vida (Liveness):**
   - Incorporar detección de parpadeo midiendo el Eye Aspect Ratio (EAR) sobre los 68 landmarks de dlib para evitar suplantaciones con fotos estáticas.
