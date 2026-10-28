"""
Configuración centralizada del sistema de control de acceso facial.

Este módulo concentra todas las constantes y parámetros configurables del sistema.
Los valores pueden ajustarse aquí sin modificar la lógica de los demás módulos.

Umbral de distancia (UMBRAL_DISTANCIA):
    face_recognition utiliza distancia euclidiana entre embeddings de 128 dimensiones.
    - Valores más bajos = mayor exigencia (menos falsos positivos, más falsos negativos).
    - Valores más altos = menor exigencia (más falsos positivos, menos falsos negativos).
    - 0.6 es el valor por defecto recomendado por la librería.
    - Para mayor seguridad en control de acceso, considerar 0.5 o inferior.
    - Este valor NO representa un porcentaje de certeza ni una probabilidad de identidad.

IMPORTANTE — Limitaciones de seguridad:
    Este sistema NO implementa detección de suplantación (anti-spoofing).
    Una fotografía o video de un residente presentado ante la cámara
    podría ser aceptado como válido. Para un despliegue en producción, se requiere integrar
    mecanismos adicionales de verificación.
"""

import os

# ── Directorio base del proyecto ──────────────────────────────────────────────
DIRECTORIO_BASE = os.path.dirname(os.path.abspath(__file__))

# ── Rutas de almacenamiento ───────────────────────────────────────────────────
DIRECTORIO_RESIDENTES = os.path.join(DIRECTORIO_BASE, "Residentes")
DIRECTORIO_DATOS = os.path.join(DIRECTORIO_BASE, "data")
ARCHIVO_RESIDENTES_DB = os.path.join(DIRECTORIO_DATOS, "residentes.json")
ARCHIVO_REGISTRO_EVENTOS = os.path.join(DIRECTORIO_BASE, "registro_accesos.csv")

# ── Motor de reconocimiento facial ────────────────────────────────────────────
# Distancia euclidiana máxima para considerar una coincidencia.
# Consultar la documentación del módulo para interpretar este valor.
UMBRAL_DISTANCIA = 0.6

# Modelo de detección de rostros:
#   "hog"  — Más rápido, funciona en CPU. Adecuado para la mayoría de los casos.
#   "cnn"  — Más preciso, requiere GPU con CUDA para rendimiento aceptable.
MODELO_DETECCION = "hog"

# ── Imágenes ──────────────────────────────────────────────────────────────────
EXTENSIONES_IMAGEN_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp")
TAMANO_MAXIMO_IMAGEN_BYTES = 10 * 1024 * 1024  # 10 MB

# ── Cámara ────────────────────────────────────────────────────────────────────
INDICE_CAMARA_DEFAULT = 0
# Cooldown en segundos entre registros consecutivos del mismo residente
# (para evitar duplicados durante monitoreo continuo en video).
TIEMPO_ENTRE_REGISTROS_SEG = 15

# ── Formato del registro de eventos ──────────────────────────────────────────
# Los campos del CSV de registro siguen este orden:
CAMPOS_CSV = [
    "fecha_hora", "residente_id", "nombre", "resultado",
    "movimiento", "distancia", "motivo"
]

# ── Nota sobre zona horaria ──────────────────────────────────────────────────
# Las fechas y horas se registran en formato ISO 8601 con la zona horaria local
# del sistema operativo. Ejemplo: 2026-10-24T10:45:00-03:00 (Argentina, ART).

