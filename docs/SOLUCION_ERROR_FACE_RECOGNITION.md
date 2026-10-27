# Solución Técnica: Error de `face_recognition_models` en Python 3.12 / 3.14

**Problema:** El script `model_trainer.py` se estancaba y mostraba repetidamente el mensaje:
```text
Please install `face_recognition_models` with this command before using `face_recognition`:
pip install git+https://github.com/ageitgey/face_recognition_models
```
incluso después de haber instalado el paquete en múltiples ocasiones.

---

## 1. Causa Raíz del Problema

La biblioteca `face_recognition` tiene la siguiente estructura en su archivo `face_recognition/api.py`:

```python
try:
    import face_recognition_models
except Exception:
    print("Please install `face_recognition_models` with this command before using `face_recognition`:\n")
    print("pip install git+https://github.com/ageitgey/face_recognition_models")
    quit()
```

Nótese que captura **cualquier `Exception`** genérica y asume que el paquete no está instalado.

Al inspeccionar directamente la importación de `face_recognition_models`:
```bash
python -c "import face_recognition_models"
```
Se reveló el verdadero error subyacente:
```text
Traceback (most recent call last):
  File "site-packages\face_recognition_models\__init__.py", line 7, in <module>
    from pkg_resources import resource_filename
ModuleNotFoundError: No module named 'pkg_resources'
```

### ¿Por qué faltaba `pkg_resources`?
- `pkg_resources` era un módulo legado distribuido históricamente con la biblioteca `setuptools`.
- A partir de las versiones de `setuptools >= 70` (y acentuado en `setuptools 84` presente en entornos Python 3.14), `pkg_resources` fue formalmente **retirado/deprecado**.
- En consecuencia, aunque `face_recognition_models` estaba físicamente instalado en `site-packages`, su `import` fallaba en la línea 7 al no encontrar `pkg_resources`. Esto disparaba el bloque `except Exception` de `face_recognition`, produciendo el mensaje engañoso.

---

## 2. Solución Aplicada

Se implementó una solución en tres niveles para garantizar estabilidad total:

### Nivel 1: Downgrade Controlado de `setuptools`
Se instaló una versión de `setuptools` compatible que aún incluye `pkg_resources`:
```powershell
pip install "setuptools<70"
```
(Específicamente quedó instalada la versión estable `setuptools==69.5.1`).

### Nivel 2: Parche Preventivo en `face_recognition_models`
En `face_recognition_models/__init__.py`, se agregó un mecanismo de respaldo automático (fallback):
```python
try:
    from pkg_resources import resource_filename
except ImportError:
    import os
    def resource_filename(package_or_requirement, resource_name):
        return os.path.join(os.path.dirname(__file__), resource_name)
```
De esta forma, incluso si en el futuro `setuptools` se actualiza o no está presente, `face_recognition_models` utilizará la ruta del sistema de archivos sin fallar.

### Nivel 3: Fijación en `requirements.txt`
Se añadió la regla explícita en `requirements.txt`:
```txt
setuptools>=65.0.0,<70.0.0
```
asegurando que cualquier reinstalación del entorno virtual respete esta compatibilidad.

---

## 3. Verificación de Funcionamiento

Tras la solución, el comando de verificación:
```powershell
.\.venv\Scripts\python.exe -c "import face_recognition; print('Éxito')"
```
retornó de manera instantánea:
```text
Éxito
```
Posteriormente, el entrenamiento procesó las 112 imágenes del dataset y generó el archivo `modelo_biometrico.pkl` sin interrupciones.

