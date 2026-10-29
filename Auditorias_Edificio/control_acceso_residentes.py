"""
Punto de entrada y terminal del sistema de control de acceso.

Coordina la cámara web, la prueba de vida (liveness detection anti-spoofing),
la identificación facial biométrica, la autorización del residente y el
registro de eventos de auditoría en CSV.
"""

import os
import sys
import time
import warnings
from typing import Optional, Tuple

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="face_recognition_models",
)

import cv2
import numpy as np

from camera import CameraManager
from config import (
    UMBRAL_DISTANCIA,
    TIEMPO_ENTRE_REGISTROS_SEG,
    ARCHIVO_REGISTRO_EVENTOS,
)
from liveness import PruebaVida
from resident_manager import ResidenteManager
from event_logger import EventLogger
from access_control import AccessController
from face_engine import (
    detectar_rostros,
    codificar_rostro,
    identificar_1aN,
)
from models import TipoMovimiento


# ── Funciones de soporte y compatibilidad ─────────────────────────────────────

def abrir_camara(indice: int = 0) -> cv2.VideoCapture:
    """Abre y configura el dispositivo de captura de video."""
    captura = cv2.VideoCapture(indice, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not captura.isOpened():
        captura = cv2.VideoCapture(indice)
    return captura


def inicializar_archivo_auditoria(ruta_csv: Optional[str] = None):
    """Garantiza la inicialización del archivo de auditoría mediante EventLogger."""
    ruta = ruta_csv or ARCHIVO_REGISTRO_EVENTOS
    EventLogger(ruta)


def determinar_siguiente_movimiento(nombre_o_id: str, ruta_csv: Optional[str] = None) -> str:
    """
    Determina si el siguiente movimiento es ENTRADA o SALIDA para una persona.
    """
    ruta = ruta_csv or ARCHIVO_REGISTRO_EVENTOS
    logger = EventLogger(ruta)
    # Buscar por id o por nombre en el historial
    historial = logger.obtener_historial()
    for evento in reversed(historial):
        if evento.nombre.lower() == nombre_o_id.lower() or evento.residente_id == nombre_o_id:
            if evento.movimiento == TipoMovimiento.ENTRADA.value:
                return TipoMovimiento.SALIDA.value
            elif evento.movimiento == TipoMovimiento.SALIDA.value:
                return TipoMovimiento.ENTRADA.value
    return TipoMovimiento.ENTRADA.value


def registrar_acceso_auditoria(nombre: str, ruta_csv: Optional[str] = None) -> str:
    """
    Registra un acceso rápido para un residente autorizado alternando ENTRADA / SALIDA.
    """
    ruta = ruta_csv or ARCHIVO_REGISTRO_EVENTOS
    logger = EventLogger(ruta)
    movimiento_str = determinar_siguiente_movimiento(nombre, ruta)
    mov = TipoMovimiento.ENTRADA if movimiento_str == "ENTRADA" else TipoMovimiento.SALIDA
    logger.registrar_acceso_autorizado(
        residente_id="",
        nombre=nombre,
        movimiento=mov,
        distancia=0.0,
    )
    return movimiento_str


def dibujar_texto(frame, texto: str, color=(255, 255, 255), y: int = 35):
    """Muestra un mensaje superior en la ventana de video."""
    cv2.putText(
        frame,
        texto,
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def dibujar_rostros(frame, ubicaciones, color=(0, 220, 255)):
    """Dibuja los rectángulos de los rostros detectados."""
    for arriba, derecha, abajo, izquierda in ubicaciones:
        cv2.rectangle(
            frame,
            (izquierda, arriba),
            (derecha, abajo),
            color,
            2,
        )


def dibujar_interfaz_acceso(imagen, ubicacion_cara, texto_etiqueta: str, autorizado=None):
    """
    Dibuja el recuadro y la etiqueta identificatoria sobre el rostro.
    Verde = Autorizado, Rojo = Denegado, Naranja = Evaluando liveness.
    """
    arriba, derecha, abajo, izquierda = ubicacion_cara
    if autorizado is True:
        color = (0, 200, 0)
    elif autorizado is False:
        color = (0, 0, 220)
    else:
        color = (0, 165, 255)

    cv2.rectangle(imagen, (izquierda, arriba), (derecha, abajo), color, 2)
    cv2.rectangle(imagen, (izquierda, abajo - 35), (derecha, abajo), color, cv2.FILLED)
    cv2.putText(
        imagen,
        texto_etiqueta,
        (izquierda + 6, abajo - 8),
        cv2.FONT_HERSHEY_DUPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


# ── Modos de ejecución de terminal ───────────────────────────────────────────

def ejecutar_modo_captura_unica(
    residente_manager: ResidenteManager,
    event_logger: EventLogger,
    access_controller: AccessController,
):
    """
    Modo instantáneo: Captura un único fotograma e identifica los rostros presentes.
    """
    print("\n[INFO] Iniciando cámara para captura instantánea...")
    encodings, residentes = residente_manager.cargar_galeria()

    if not encodings:
        print("[ADVERTENCIA] No hay residentes cargados en la galería.")
        return

    with CameraManager() as camara:
        if not camara.esta_abierta():
            print("[ERROR] No se pudo abrir la cámara.")
            return

        # Descartar primeros frames para auto-exposición de la cámara
        for _ in range(5):
            camara.leer_frame()

        exito, frame = camara.leer_frame()

    if not exito or frame is None:
        print("[ERROR] No se pudo obtener la captura de video.")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ubicaciones = detectar_rostros(frame_rgb)

    if not ubicaciones:
        print("[INFO] No se detectó ningún rostro en la captura.")
        dibujar_texto(frame, "No se detectaron rostros", (0, 0, 255), 45)
    else:
        for ubicacion in ubicaciones:
            encoding_captura = codificar_rostro(frame_rgb, ubicacion)
            if encoding_captura is not None:
                resultado = identificar_1aN(
                    encoding_captura,
                    encodings,
                    residentes,
                    umbral=UMBRAL_DISTANCIA,
                )
                autorizado, mensaje, evento = access_controller.procesar_identificacion(resultado)
                etiqueta = resultado.residente.nombre if (autorizado and resultado.residente) else "DENEGADO"
                dibujar_interfaz_acceso(frame, ubicacion, etiqueta, autorizado)
                if evento:
                    print(f"[AUDITORÍA] {evento.nombre} -> {evento.resultado} ({evento.movimiento})")
            else:
                dibujar_interfaz_acceso(frame, ubicacion, "Ilegible", False)

    cv2.imshow("Captura Única - Terminal de Acceso", frame)
    print("Presione cualquier tecla en la ventana de imagen para continuar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ejecutar_modo_tiempo_real(
    residente_manager: ResidenteManager,
    event_logger: EventLogger,
    access_controller: AccessController,
):
    """
    Ejecuta el flujo completo de acceso en tiempo real con Prueba de Vida.
    """
    encodings, residentes = residente_manager.cargar_galeria()

    if not encodings:
        print("[ADVERTENCIA] No hay imágenes faciales válidas en la galería de residentes.")
        print("Registrá residentes con imágenes válidas antes de probar el reconocimiento.")
        return

    print(f"[INFO] Codificaciones biométricas cargadas: {len(encodings)}")
    print("[INFO] Iniciando cámara con Prueba de Vida. Presioná Q o ESC para salir.\n")

    prueba_vida = None
    esperando_retirada = False

    mensaje_resultado = None
    color_resultado = (255, 255, 255)
    mostrar_resultado_hasta = 0.0

    with CameraManager() as camara:
        if not camara.esta_abierta():
            print("[ERROR] No se pudo abrir la cámara.")
            return

        while True:
            exito, frame = camara.leer_frame()

            if not exito or frame is None:
                print("[ERROR] No se pudo leer la cámara.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ubicaciones = detectar_rostros(frame_rgb)

            if esperando_retirada:
                dibujar_rostros(frame, ubicaciones)

                if time.monotonic() < mostrar_resultado_hasta:
                    if mensaje_resultado is not None:
                        dibujar_texto(frame, mensaje_resultado, color_resultado, 60)
                else:
                    dibujar_texto(
                        frame,
                        "Por favor, retírese para el siguiente acceso",
                        (0, 220, 255),
                        60,
                    )
                    if not ubicaciones:
                        esperando_retirada = False
                        prueba_vida = None
                        mensaje_resultado = None

                cv2.imshow("Control de Acceso Facial - Terminal", frame)

            elif not ubicaciones:
                prueba_vida = None
                dibujar_texto(frame, "Esperando persona frente a la cámara...", (255, 255, 255), 60)
                cv2.imshow("Control de Acceso Facial - Terminal", frame)

            elif len(ubicaciones) > 1:
                prueba_vida = None
                dibujar_rostros(frame, ubicaciones, (0, 0, 255))
                dibujar_texto(
                    frame,
                    "Debe haber una sola persona frente a la cámara",
                    (0, 0, 255),
                    60,
                )
                cv2.imshow("Control de Acceso Facial - Terminal", frame)

            else:
                if prueba_vida is None:
                    prueba_vida = PruebaVida()

                estado, caras_vida, ear, mar = prueba_vida.evaluar(frame)
                dibujar_rostros(frame, caras_vida)

                dibujar_texto(frame, prueba_vida.obtener_instruccion(), (0, 220, 255), 60)
                dibujar_texto(frame, estado, (255, 255, 255), 90)

                if ear is not None:
                    dibujar_texto(frame, f"EAR (Ojos): {ear:.3f}", (255, 255, 255), 120)
                if mar is not None:
                    dibujar_texto(frame, f"MAR (Boca): {mar:.3f}", (255, 255, 255), 150)

                if prueba_vida.superada:
                    print("[OK] Prueba de vida superada.")
                    encoding_captura = codificar_rostro(frame_rgb)

                    if encoding_captura is None:
                        mensaje_resultado = "No se pudo codificar el rostro"
                        color_resultado = (0, 0, 255)
                    else:
                        resultado = identificar_1aN(
                            encoding_captura,
                            encodings,
                            residentes,
                            umbral=UMBRAL_DISTANCIA,
                        )

                        autorizado, mensaje_res, evento = access_controller.procesar_identificacion(resultado)
                        color_resultado = (0, 200, 0) if autorizado else (0, 0, 255)

                        if resultado.identificado and resultado.residente:
                            mov = evento.movimiento if evento else "ACCESO"
                            mensaje_resultado = f"AUTORIZADO: {resultado.residente.nombre} ({mov})"
                        else:
                            mensaje_resultado = "ACCESO DENEGADO - Persona no registrada"

                        if evento is not None:
                            print(f"[AUDITORÍA] {evento.nombre} -> {evento.resultado} ({evento.movimiento})")

                    mostrar_resultado_hasta = time.monotonic() + 3.0
                    esperando_retirada = True

                elif prueba_vida.fallida:
                    mensaje_resultado = "Acceso denegado: prueba de vida fallida"
                    color_resultado = (0, 0, 255)
                    event_logger.registrar_acceso_denegado(motivo=mensaje_resultado)
                    mostrar_resultado_hasta = time.monotonic() + 3.0
                    esperando_retirada = True

                cv2.imshow("Control de Acceso Facial - Terminal", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q") or tecla == 27:
                break

    cv2.destroyAllWindows()


def main():
    """Punto de entrada de terminal con selección interactiva."""
    print("=" * 70)
    print("  SISTEMA DE CONTROL DE ACCESO FACIAL Y AUDITORÍA RESIDENCIAL")
    print("=" * 70)

    residente_manager = ResidenteManager()
    event_logger = EventLogger()
    access_controller = AccessController(
        event_logger=event_logger,
        tiempo_cooldown=TIEMPO_ENTRE_REGISTROS_SEG,
    )

    residentes = residente_manager.listar()
    print(f"\n[INFO] Residentes registrados en el sistema: {len(residentes)}")
    for r in residentes:
        st = "AUTORIZADO" if r.autorizado else "NO AUTORIZADO"
        print(f"  • {r.nombre} ({st})")

    print("\nModos de operación disponibles:")
    print("  [1] Monitoreo en Tiempo Real con Prueba de Vida (Recomendado)")
    print("  [2] Captura Instantánea (Foto rápida sin prueba de vida)")
    opcion = input("\nSeleccione opción [1 o 2, Enter=1]: ").strip()

    if opcion == "2":
        ejecutar_modo_captura_unica(residente_manager, event_logger, access_controller)
    else:
        ejecutar_modo_tiempo_real(residente_manager, event_logger, access_controller)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Programa finalizado por el usuario.")
    except Exception as error:
        print(f"\n[ERROR] {error}")
