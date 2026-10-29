"""
Gestión de la cámara web.

Encapsula la inicialización, captura y liberación de la cámara web,
incluyendo manejo de errores, prevención de múltiples instancias y
liberación segura de recursos.
"""

import cv2
from typing import Optional, Tuple

import numpy as np

from config import INDICE_CAMARA_DEFAULT


class CameraManager:
    """
    Administrador de la cámara web del punto de acceso.

    Soporta uso como context manager (with) para garantizar la liberación
    de recursos incluso en caso de error.

    Ejemplo:
        with CameraManager() as cam:
            if cam.esta_abierta():
                exito, frame = cam.leer_frame()
    """

    def __init__(self, indice: int = INDICE_CAMARA_DEFAULT):
        """
        Args:
            indice: Índice del dispositivo de cámara (0 = cámara principal).
        """
        self.indice = indice
        self._captura: Optional[cv2.VideoCapture] = None

    def __enter__(self):
        self.abrir()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cerrar()
        return False

    def abrir(self) -> bool:
        """
        Abre la cámara web.

        En Windows, intenta primero con el backend DirectShow (CAP_DSHOW)
        para mejor compatibilidad, con fallback al backend por defecto.

        Returns:
            True si la cámara se abrió correctamente.
        """
        if self._captura is not None and self._captura.isOpened():
            return True  # Ya está abierta

        # Intentar con DirectShow (Windows)
        self._captura = cv2.VideoCapture(self.indice, cv2.CAP_DSHOW)
        if self._captura.isOpened():
            return True

        # Fallback al backend por defecto
        self._captura.release()
        self._captura = cv2.VideoCapture(self.indice)

        if not self._captura.isOpened():
            self._captura.release()
            self._captura = None
            return False

        return True

    def esta_abierta(self) -> bool:
        """Retorna True si la cámara está abierta y operativa."""
        return self._captura is not None and self._captura.isOpened()

    def leer_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Captura un fotograma de la cámara.

        Returns:
            Tupla (exito, frame_bgr). El frame está en formato BGR (OpenCV).
            Si falla, retorna (False, None).
        """
        if not self.esta_abierta():
            return False, None

        exito, frame = self._captura.read()
        if not exito or frame is None:
            return False, None

        return True, frame

    def capturar_foto(self) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Abre la cámara, captura una foto y la cierra.

        Función de conveniencia para el modo de captura única.

        Returns:
            Tupla (exito, imagen_bgr, mensaje).
        """
        if not self.abrir():
            return False, None, (
                "No se pudo abrir la cámara web. Verifique que:\n"
                "  - La cámara esté conectada.\n"
                "  - No esté siendo usada por otra aplicación.\n"
                "  - Los permisos de cámara estén habilitados en Windows."
            )

        exito, frame = self.leer_frame()
        self.cerrar()

        if not exito or frame is None:
            return False, None, "No se pudo capturar la imagen desde la cámara web."

        return True, frame, "Imagen capturada correctamente."

    def cerrar(self):
        """
        Libera los recursos de la cámara.

        Esta función es segura de llamar múltiples veces o aunque la cámara
        no esté abierta.
        """
        if self._captura is not None:
            try:
                self._captura.release()
            except Exception:
                pass
            self._captura = None

