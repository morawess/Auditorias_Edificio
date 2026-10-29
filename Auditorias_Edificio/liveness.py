"""
Prueba de vida mediante un desafio aleatorio:
- Parpadeo: utiliza EAR.
- Apertura de boca: utiliza MAR.

Esta clase no identifica residentes ni autoriza accesos.
"""

import random
import time

import cv2
import face_recognition
import numpy as np


TIEMPO_LIMITE = 10.0

UMBRAL_OJO_CERRADO = 0.20
UMBRAL_OJO_ABIERTO = 0.24
FOTOGRAMAS_OJO_CERRADO = 2

UMBRAL_BOCA_ABIERTA = 0.54
FOTOGRAMAS_BOCA_ABIERTA = 2


class PruebaVida:
    """Selecciona y evalua un desafio de vida."""

    DESAFIOS = ("parpadeo", "boca")

    def __init__(self, tiempo_limite=TIEMPO_LIMITE):
        self.tiempo_limite = tiempo_limite
        self.reiniciar()

    def reiniciar(self):
        """Inicia un intento nuevo y elige un desafio aleatorio."""
        self.inicio = time.monotonic()
        self.desafio = random.choice(self.DESAFIOS)

        self.superada = False
        self.fallida = False

        self.fotogramas_ojos_cerrados = 0
        self.ojos_cerrados_detectados = False
        self.fotogramas_boca_abierta = 0

    def obtener_instruccion(self):
        """Devuelve la instruccion correspondiente al desafio."""
        if self.desafio == "parpadeo":
            return "Desafio: parpadea y vuelve a abrir los ojos"

        return "Desafio: abre la boca"

    @staticmethod
    def calcular_ear(puntos_ojo):
        """Calcula la relacion de apertura de un ojo."""
        puntos = np.array(puntos_ojo, dtype=float)

        if len(puntos) != 6:
            return None

        vertical_1 = np.linalg.norm(puntos[1] - puntos[5])
        vertical_2 = np.linalg.norm(puntos[2] - puntos[4])
        horizontal = np.linalg.norm(puntos[0] - puntos[3])

        if horizontal == 0:
            return None

        return float(
            (vertical_1 + vertical_2) / (2.0 * horizontal)
        )

    @classmethod
    def calcular_ear_promedio(cls, landmarks):
        """Calcula el EAR promedio de ambos ojos."""
        ojo_izquierdo = landmarks.get("left_eye")
        ojo_derecho = landmarks.get("right_eye")

        if not ojo_izquierdo or not ojo_derecho:
            return None

        ear_izquierdo = cls.calcular_ear(ojo_izquierdo)
        ear_derecho = cls.calcular_ear(ojo_derecho)

        if ear_izquierdo is None or ear_derecho is None:
            return None

        return (ear_izquierdo + ear_derecho) / 2.0

    @staticmethod
    def calcular_mar(landmarks):
        """Calcula una relacion aproximada de apertura de la boca."""
        labio_superior = landmarks.get("top_lip")
        labio_inferior = landmarks.get("bottom_lip")

        if (
            not labio_superior
            or not labio_inferior
            or len(labio_superior) < 7
            or len(labio_inferior) < 10
        ):
            return None

        puntos_superiores = np.array(labio_superior, dtype=float)
        puntos_inferiores = np.array(labio_inferior, dtype=float)

        ancho = np.linalg.norm(
            puntos_superiores[0] - puntos_superiores[6]
        )

        if ancho == 0:
            return None

        vertical_1 = np.linalg.norm(
            puntos_superiores[3] - puntos_inferiores[9]
        )
        vertical_2 = np.linalg.norm(
            puntos_superiores[4] - puntos_inferiores[8]
        )
        vertical_3 = np.linalg.norm(
            puntos_superiores[5] - puntos_inferiores[7]
        )

        return float(
            (vertical_1 + vertical_2 + vertical_3)
            / (3.0 * ancho)
        )

    def evaluar(self, frame_bgr):
        """
        Evalua un fotograma.

        Retorna:
        (estado, ubicaciones, ear, mar)
        """
        if self.superada:
            return "PRUEBA DE VIDA SUPERADA", [], None, None

        if self.fallida:
            return "PRUEBA DE VIDA FALLIDA", [], None, None

        transcurrido = time.monotonic() - self.inicio

        if transcurrido >= self.tiempo_limite:
            self.fallida = True
            return "Tiempo agotado: prueba fallida", [], None, None

        frame_rgb = cv2.cvtColor(
            frame_bgr, cv2.COLOR_BGR2RGB
        )

        ubicaciones = face_recognition.face_locations(
            frame_rgb, model="hog"
        )

        if len(ubicaciones) != 1:
            self._reiniciar_secuencia()

            if len(ubicaciones) == 0:
                return "No se detecta un rostro", ubicaciones, None, None

            return "Debe haber una sola persona", ubicaciones, None, None

        landmarks_lista = face_recognition.face_landmarks(
            frame_rgb,
            face_locations=ubicaciones,
            model="large",
        )

        if not landmarks_lista:
            return (
                "No se pudieron analizar los puntos faciales",
                ubicaciones,
                None,
                None,
            )

        landmarks = landmarks_lista[0]
        ear = self.calcular_ear_promedio(landmarks)
        mar = self.calcular_mar(landmarks)

        if self.desafio == "parpadeo":
            estado = self._evaluar_parpadeo(ear)

        else:
            estado = self._evaluar_boca(mar)

        restante = max(
            0.0,
            self.tiempo_limite
            - (time.monotonic() - self.inicio),
        )

        if self.superada:
            estado = "Desafio superado"

        return (
            f"{estado} ({restante:.1f} s)",
            ubicaciones,
            ear,
            mar,
        )

    def _evaluar_parpadeo(self, ear):
        """Aprueba cuando detecta cierre y posterior apertura."""
        if ear is None:
            return "No se pudieron analizar los ojos"

        if ear < UMBRAL_OJO_CERRADO:
            self.fotogramas_ojos_cerrados += 1

            if self.fotogramas_ojos_cerrados >= FOTOGRAMAS_OJO_CERRADO:
                self.ojos_cerrados_detectados = True

            return "Cierre detectado; abre los ojos"

        if ear > UMBRAL_OJO_ABIERTO:
            if self.ojos_cerrados_detectados:
                self.superada = True
                return "Parpadeo detectado"

            self.fotogramas_ojos_cerrados = 0
            return "Parpadea y abre los ojos"

        return "Parpadea naturalmente"

    def _evaluar_boca(self, mar):
        """Aprueba si detecta la boca abierta durante varios fotogramas."""
        if mar is None:
            return "No se pudo analizar la boca"

        if mar > UMBRAL_BOCA_ABIERTA:
            self.fotogramas_boca_abierta += 1

            if self.fotogramas_boca_abierta >= FOTOGRAMAS_BOCA_ABIERTA:
                self.superada = True
                return "Apertura de boca detectada"

            return "Mantene la boca abierta un instante"

        self.fotogramas_boca_abierta = 0
        return "Abre la boca"

    def _reiniciar_secuencia(self):
        """Limpia las mediciones consecutivas si se pierde el rostro."""
        self.fotogramas_ojos_cerrados = 0
        self.ojos_cerrados_detectados = False
        self.fotogramas_boca_abierta = 0
