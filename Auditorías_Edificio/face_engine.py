"""
Motor de reconocimiento facial.

Encapsula las operaciones de detección, codificación y comparación de rostros
utilizando la librería face_recognition (basada en dlib).

Manejo de espacios de color:
    - face_recognition.load_image_file() retorna imágenes en RGB.
    - cv2.imread() retorna imágenes en BGR.
    - Todas las funciones de face_recognition esperan imágenes en RGB.
    - cv2.imshow() espera imágenes en BGR.

    Este módulo trabaja internamente con RGB. Las funciones que reciben imágenes
    documentan explícitamente el espacio de color esperado.
"""

import os
from typing import List, Optional, Tuple

import numpy as np

from config import (
    EXTENSIONES_IMAGEN_VALIDAS,
    MODELO_DETECCION,
    TAMANO_MAXIMO_IMAGEN_BYTES,
    UMBRAL_DISTANCIA,
)
from models import ResultadoIdentificacion, ResultadoVerificacion, Residente


def validar_archivo_imagen(ruta: str) -> Tuple[bool, str]:
    """
    Valida que un archivo sea una imagen admitida para el sistema.

    Comprueba existencia, extensión, tamaño y que pueda cargarse correctamente.

    Args:
        ruta: Ruta absoluta o relativa al archivo de imagen.

    Returns:
        Tupla (es_valida, mensaje). Si no es válida, el mensaje explica el motivo.
    """
    if not ruta or not isinstance(ruta, str):
        return False, "La ruta de imagen está vacía o no es válida."

    if not os.path.isfile(ruta):
        return False, f"El archivo no existe: {ruta}"

    _, ext = os.path.splitext(ruta)
    if ext.lower() not in EXTENSIONES_IMAGEN_VALIDAS:
        extensiones = ", ".join(EXTENSIONES_IMAGEN_VALIDAS)
        return False, (
            f"Extensión '{ext}' no admitida. "
            f"Formatos válidos: {extensiones}"
        )

    tamano = os.path.getsize(ruta)
    if tamano == 0:
        return False, "El archivo de imagen está vacío (0 bytes)."
    if tamano > TAMANO_MAXIMO_IMAGEN_BYTES:
        mb = TAMANO_MAXIMO_IMAGEN_BYTES / (1024 * 1024)
        return False, f"El archivo excede el tamaño máximo permitido ({mb:.0f} MB)."

    return True, "Imagen válida."


def cargar_imagen_rgb(ruta: str) -> np.ndarray:
    """
    Carga una imagen desde archivo y la retorna en formato RGB.

    Utiliza face_recognition.load_image_file(), que internamente usa Pillow
    y retorna un array numpy en RGB.

    Args:
        ruta: Ruta al archivo de imagen.

    Returns:
        Array numpy con la imagen en formato RGB.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si la imagen no puede cargarse o está corrupta.
    """
    import face_recognition as fr

    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No se encontró el archivo de imagen: {ruta}")

    try:
        imagen = fr.load_image_file(ruta)
    except Exception as e:
        raise ValueError(f"No se pudo cargar la imagen '{ruta}': {e}")

    if imagen is None or imagen.size == 0:
        raise ValueError(f"La imagen cargada está vacía o corrupta: {ruta}")

    return imagen


def cargar_imagen_bgr(ruta: str) -> np.ndarray:
    """
    Carga una imagen desde archivo y la retorna en formato BGR (para OpenCV).

    Args:
        ruta: Ruta al archivo de imagen.

    Returns:
        Array numpy con la imagen en formato BGR.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si la imagen no puede cargarse.
    """
    import cv2

    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No se encontró el archivo de imagen: {ruta}")

    # En Windows, cv2.imread no soporta rutas con tildes o caracteres Unicode.
    datos = np.fromfile(ruta, dtype=np.uint8)
    imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR) if datos.size > 0 else None
    if imagen is None:
        raise ValueError(f"No se pudo leer la imagen (puede estar corrupta): {ruta}")

    return imagen


def bgr_a_rgb(imagen_bgr: np.ndarray) -> np.ndarray:
    """Convierte una imagen de BGR (OpenCV) a RGB (face_recognition)."""
    import cv2
    return cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)


def rgb_a_bgr(imagen_rgb: np.ndarray) -> np.ndarray:
    """Convierte una imagen de RGB (face_recognition) a BGR (OpenCV)."""
    import cv2
    return cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2BGR)


def detectar_rostros(
    imagen_rgb: np.ndarray,
    modelo: str = MODELO_DETECCION,
) -> List[Tuple[int, int, int, int]]:
    """
    Detecta todos los rostros presentes en una imagen RGB.

    Args:
        imagen_rgb: Imagen en formato RGB.
        modelo: Modelo de detección ("hog" o "cnn").

    Returns:
        Lista de ubicaciones (top, right, bottom, left) de cada rostro detectado.
    """
    import face_recognition as fr
    return fr.face_locations(imagen_rgb, model=modelo)


def codificar_rostro(
    imagen_rgb: np.ndarray,
    ubicacion: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[np.ndarray]:
    """
    Extrae el embedding facial (vector de 128 dimensiones) de un rostro.

    Args:
        imagen_rgb: Imagen en formato RGB.
        ubicacion: Coordenadas (top, right, bottom, left) del rostro.
                   Si es None, se detecta automáticamente el primer rostro.

    Returns:
        Array numpy de 128 dimensiones con el embedding, o None si no se
        pudo codificar.
    """
    import face_recognition as fr

    if ubicacion is not None:
        codificaciones = fr.face_encodings(imagen_rgb, [ubicacion])
    else:
        codificaciones = fr.face_encodings(imagen_rgb)

    if not codificaciones:
        return None

    return codificaciones[0]


def codificar_todos_los_rostros(
    imagen_rgb: np.ndarray,
    ubicaciones: Optional[List[Tuple[int, int, int, int]]] = None,
) -> List[np.ndarray]:
    """
    Extrae los embeddings faciales de todos los rostros en una imagen.

    Args:
        imagen_rgb: Imagen en formato RGB.
        ubicaciones: Lista de ubicaciones de rostros. Si es None, se detectan
                     automáticamente.

    Returns:
        Lista de arrays numpy de 128 dimensiones.
    """
    import face_recognition as fr

    if ubicaciones is not None:
        return fr.face_encodings(imagen_rgb, ubicaciones)
    return fr.face_encodings(imagen_rgb)


def verificar_1a1(
    encoding_a: np.ndarray,
    encoding_b: np.ndarray,
    umbral: float = UMBRAL_DISTANCIA,
) -> ResultadoVerificacion:
    """
    Verificación facial 1:1 — Determina si dos embeddings corresponden
    a la misma persona.

    La distancia euclidiana entre embeddings de 128 dimensiones se compara
    contra el umbral configurado. Un valor menor indica mayor similitud.

    NOTA: Este resultado NO es una certeza absoluta de identidad. Es una
    medida de similitud entre representaciones faciales del modelo.

    Args:
        encoding_a: Embedding del primer rostro (128D).
        encoding_b: Embedding del segundo rostro (128D).
        umbral: Distancia máxima para considerar coincidencia.

    Returns:
        ResultadoVerificacion con el resultado de la comparación.
    """
    import face_recognition as fr

    distancia = float(fr.face_distance([encoding_a], encoding_b)[0])
    coincide = distancia <= umbral

    if coincide:
        mensaje = (
            f"Coincidencia detectada (distancia: {distancia:.4f}, "
            f"umbral: {umbral}). Los rostros presentan alta similitud."
        )
    else:
        mensaje = (
            f"No coinciden (distancia: {distancia:.4f}, umbral: {umbral}). "
            f"Los rostros no presentan similitud suficiente."
        )

    return ResultadoVerificacion(
        coincide=coincide,
        distancia=distancia,
        mensaje=mensaje,
    )


def identificar_1aN(
    encoding_captura: np.ndarray,
    galeria_encodings: List[np.ndarray],
    galeria_residentes: List[Residente],
    umbral: float = UMBRAL_DISTANCIA,
) -> ResultadoIdentificacion:
    """
    Identificación facial 1:N — Busca a qué residente registrado corresponde
    un rostro capturado.

    Compara el embedding capturado contra todos los embeddings de la galería
    y selecciona el candidato con menor distancia, siempre que esté dentro
    del umbral.

    IMPORTANTE: Encontrar al candidato más parecido NO confirma su identidad.
    Si ningún candidato está dentro del umbral, el resultado es
    "persona no identificada".

    Args:
        encoding_captura: Embedding del rostro capturado (128D).
        galeria_encodings: Lista de embeddings de residentes registrados.
        galeria_residentes: Lista de objetos Residente correspondientes.
        umbral: Distancia máxima para considerar coincidencia.

    Returns:
        ResultadoIdentificacion con el candidato identificado o no.
    """
    if not galeria_encodings or not galeria_residentes:
        return ResultadoIdentificacion(
            identificado=False,
            mensaje="No hay residentes registrados en la galería para comparar.",
        )

    import face_recognition as fr

    distancias = fr.face_distance(galeria_encodings, encoding_captura)
    indice_minimo = int(np.argmin(distancias))
    distancia_minima = float(distancias[indice_minimo])

    if distancia_minima <= umbral:
        residente = galeria_residentes[indice_minimo]
        return ResultadoIdentificacion(
            identificado=True,
            residente=residente,
            distancia=distancia_minima,
            mensaje=(
                f"Residente identificado: {residente.nombre} "
                f"(distancia: {distancia_minima:.4f})"
            ),
        )
    else:
        return ResultadoIdentificacion(
            identificado=False,
            distancia=distancia_minima,
            mensaje=(
                f"Persona no identificada. La menor distancia fue "
                f"{distancia_minima:.4f} (umbral: {umbral}). No se alcanzó "
                f"el criterio de coincidencia."
            ),
        )


def detectar_y_codificar_imagen(
    ruta: str,
    nombre_referencia: str = "Imagen",
) -> Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]:
    """
    Carga una imagen, detecta el rostro principal y extrae su embedding.

    Función de conveniencia que combina carga, detección y codificación
    para el caso más común: una imagen con un rostro esperado.

    Args:
        ruta: Ruta al archivo de imagen.
        nombre_referencia: Etiqueta para mensajes de error.

    Returns:
        Tupla (ubicacion_rostro, encoding, imagen_rgb).

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si no se detecta un rostro o no se puede codificar.
    """
    valida, msg = validar_archivo_imagen(ruta)
    if not valida:
        raise ValueError(f"Imagen no válida ({nombre_referencia}): {msg}")

    imagen_rgb = cargar_imagen_rgb(ruta)
    ubicaciones = detectar_rostros(imagen_rgb)

    if not ubicaciones:
        raise ValueError(
            f"No se detectó ningún rostro en '{nombre_referencia}' ({os.path.basename(ruta)}). "
            f"Verifique que la imagen contenga un rostro claramente visible."
        )

    if len(ubicaciones) > 1:
        print(
            f"[ADVERTENCIA] Se detectaron {len(ubicaciones)} rostros en "
            f"'{nombre_referencia}'. Se utilizará el primero."
        )

    ubicacion = ubicaciones[0]
    encoding = codificar_rostro(imagen_rgb, ubicacion)

    if encoding is None:
        raise ValueError(
            f"No se pudieron codificar las características faciales en "
            f"'{nombre_referencia}' ({os.path.basename(ruta)})."
        )

    return ubicacion, encoding, imagen_rgb

