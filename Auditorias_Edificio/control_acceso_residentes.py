"""
Punto de entrada del sistema de control de acceso.

Coordina la cámara, la prueba de vida, la identificación facial,
la autorización del residente y el registro de eventos.
"""

import cv2
import warnings
import time

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="face_recognition_models",
)

from camera import CameraManager
from config import (
    UMBRAL_DISTANCIA,
    TIEMPO_ENTRE_REGISTROS_SEG,
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


def dibujar_texto(frame, texto, color=(255, 255, 255), y=35):
    """Muestra un mensaje en la ventana de video."""
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


def ejecutar_modo_tiempo_real(
    residente_manager,
    event_logger,
    access_controller,
):
    """
    Ejecuta el flujo completo de acceso en tiempo real.
    """
    encodings, residentes = residente_manager.cargar_galeria()

    if not encodings:
        print(
            "[ADVERTENCIA] No hay imágenes faciales válidas "
            "en la galería de residentes."
        )
        print(
            "Registrá residentes con imágenes válidas antes "
            "de probar el reconocimiento."
        )
        return

    print(f"[INFO] Codificaciones cargadas: {len(encodings)}")
    print("[INFO] Iniciando cámara. Presioná Q o ESC para salir.")

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

            # Detectamos rostros para controlar el inicio y el fin
            # de cada intento de acceso.
            ubicaciones = detectar_rostros(frame_rgb)

            if esperando_retirada:
                dibujar_rostros(frame, ubicaciones)

                if time.monotonic() < mostrar_resultado_hasta:
                    # Mostrar el resultado durante tres segundos.
                    if mensaje_resultado is not None:
                        dibujar_texto(
                            frame,
                            mensaje_resultado,
                            color_resultado,
                            60,
                        )

                else:
                    # Una vez mostrado el resultado, esperar que
                    # la persona se retire antes de iniciar otro intento.
                    dibujar_texto(
                        frame,
                        "Retirese de la camara para el siguiente intento",
                        (0, 220, 255),
                        60,
                    )

                    if not ubicaciones:
                        esperando_retirada = False
                        prueba_vida = None
                        mensaje_resultado = None

                cv2.imshow("Control de acceso", frame)

            elif not ubicaciones:
                # No iniciar el tiempo límite hasta que haya una persona.
                prueba_vida = None
                dibujar_texto(
                    frame,
                    "Esperando una persona...",
                    (255, 255, 255),
                    60,
                )
                cv2.imshow("Control de acceso", frame)

            elif len(ubicaciones) > 1:
                # Para este flujo se requiere una sola persona.
                prueba_vida = None
                dibujar_rostros(frame, ubicaciones, (0, 0, 255))
                dibujar_texto(
                    frame,
                    "Debe haber una sola persona frente a la camara",
                    (0, 0, 255),
                    60,
                )
                cv2.imshow("Control de acceso", frame)

            else:
                # Crear el desafío solo al comenzar un intento.
                if prueba_vida is None:
                    prueba_vida = PruebaVida()

                estado, caras_vida, ear, mar = prueba_vida.evaluar(frame)

                dibujar_rostros(frame, caras_vida)

                dibujar_texto(
                    frame,
                    prueba_vida.obtener_instruccion(),
                    (0, 220, 255),
                    60,
                )
                dibujar_texto(frame, estado, (255, 255, 255), 90)

                if ear is not None:
                    dibujar_texto(
                        frame,
                        f"EAR: {ear:.3f}",
                        (255, 255, 255),
                        120,
                    )

                if mar is not None:
                    dibujar_texto(
                        frame,
                        f"MAR: {mar:.3f}",
                        (255, 255, 255),
                        150,
                    )

                if prueba_vida.superada:
                    print("[OK] Prueba de vida superada.")

                    # Solo se identifica después de superar el desafío.
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

                        autorizado, mensaje, evento = (
                            access_controller.procesar_identificacion(
                                resultado
                            )
                        )

                        color_resultado = (
                            (0, 180, 0)
                            if autorizado
                            else (0, 0, 255)
                        )

                        mensaje_resultado = mensaje

                        if evento is not None:
                            print(f"[EVENTO] {evento}")

                    # Mostrar el resultado durante tres segundos.
                    mostrar_resultado_hasta = time.monotonic() + 3.0

                    # No procesar repetidamente el mismo rostro.
                    esperando_retirada = True

                elif prueba_vida.fallida:
                    mensaje_resultado = (
                        "Acceso denegado: prueba de vida fallida"
                    )
                    color_resultado = (0, 0, 255)

                    event_logger.registrar_acceso_denegado(
                        motivo=mensaje_resultado
                    )
                    mostrar_resultado_hasta = time.monotonic() + 3.0
                    esperando_retirada = True

                cv2.imshow("Control de acceso", frame)

            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q") or tecla == 27:
                break

    cv2.destroyAllWindows()


def main():
    """Inicializa los componentes y ejecuta el sistema."""
    print("=" * 60)
    print(" SISTEMA DE CONTROL DE ACCESO FACIAL")
    print("=" * 60)

    residente_manager = ResidenteManager()
    event_logger = EventLogger()

    access_controller = AccessController(
        event_logger=event_logger,
        tiempo_cooldown=TIEMPO_ENTRE_REGISTROS_SEG,
    )

    residentes = residente_manager.listar()

    print(f"[INFO] Residentes registrados: {len(residentes)}")

    for residente in residentes:
        estado = (
            "AUTORIZADO"
            if residente.autorizado
            else "NO AUTORIZADO"
        )
        print(f" - {residente.nombre}: {estado}")

    ejecutar_modo_tiempo_real(
        residente_manager,
        event_logger,
        access_controller,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Programa interrumpido por el usuario.")
    except Exception as error:
        print(f"\n[ERROR DEL SISTEMA] {error}")