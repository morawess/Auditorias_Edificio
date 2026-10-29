"""
Sistema de Acceso sin Llaves y Auditoría para Residentes
---------------------------------------------------------------------------------------------
Versión Fusionada: Mantiene la interfaz gráfica y el menú original, 
integrando la nueva lógica modular (Prueba de Vida, Event Logger, Access Controller).
"""

import cv2
import warnings
import time
import os

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="face_recognition_models",
)

# Importaciones lógicas modulares (arquitectura del compañero)
from camera import CameraManager
from config import UMBRAL_DISTANCIA, TIEMPO_ENTRE_REGISTROS_SEG
from liveness import PruebaVida
from resident_manager import ResidenteManager
from event_logger import EventLogger
from access_control import AccessController
from face_engine import (
    detectar_rostros,
    codificar_rostro,
    identificar_1aN,
)

def dibujar_interfaz_acceso(imagen, ubicacion_cara, texto_etiqueta: str, autorizado=None):
    """
    Dibuja el rectángulo delimitador alrededor del rostro detectado y el cartel
    inferior. (Diseño de interfaz original).
    """
    # Desempaquetado (arriba, derecha, abajo, izquierda)
    y1, x2, y2, x1 = ubicacion_cara
    
    # Lógica de colores: Verde (Autorizado), Rojo (Denegado), Naranja (Evaluando liveness)
    if autorizado is True:
        color = (0, 200, 0)
    elif autorizado is False:
        color = (0, 0, 220)
    else:
        color = (0, 165, 255) # Naranja BGR

    # Rectángulo principal alrededor del rostro
    cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 2)
    # Rectángulo relleno inferior para el texto
    cv2.rectangle(imagen, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
    # Texto con el nombre del residente, estado o instrucción
    cv2.putText(
        imagen,
        texto_etiqueta,
        (x1 + 6, y2 - 8),
        cv2.FONT_HERSHEY_COMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

def dibujar_texto_hud(frame, texto, color=(255, 255, 255), y=35):
    """Muestra información secundaria en la esquina de la pantalla (EAR, MAR)."""
    cv2.putText(
        frame, texto, (15, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA
    )

def ejecutar_modo_captura_unica(residente_manager, event_logger, access_controller):
    """Modo original de captura instantánea, adaptado a los nuevos managers."""
    print("\nIniciando cámara web para captura instantánea (Sin prueba de vida)...")
    encodings, residentes = residente_manager.cargar_galeria()
    
    with CameraManager() as camara:
        if not camara.esta_abierta():
            print("[ERROR] No se pudo abrir la cámara.")
            return
        exito, frame = camara.leer_frame()
        
    if not exito or frame is None:
        print("[ERROR] No se pudo tomar la captura.")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ubicaciones = detectar_rostros(frame_rgb)

    if not ubicaciones:
        print("[INFO] No se detectaron rostros.")
        dibujar_texto_hud(frame, "Sin rostros detectados", (0, 165, 255), 45)
    else:
        for ubicacion in ubicaciones:
            encoding_captura = codificar_rostro(frame_rgb)
            if encoding_captura is not None:
                resultado = identificar_1aN(encoding_captura, encodings, residentes, umbral=UMBRAL_DISTANCIA)
                autorizado, mensaje, evento = access_controller.procesar_identificacion(resultado)
                dibujar_interfaz_acceso(frame, ubicacion, mensaje, autorizado)
                if evento: print(f"[EVENTO] {evento}")
            else:
                dibujar_interfaz_acceso(frame, ubicacion, "Rostro ilegible", False)

    cv2.imshow("Terminal de Acceso - Captura Web", frame)
    print("Presione cualquier tecla en la ventana de imagen para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ejecutar_modo_tiempo_real(residente_manager, event_logger, access_controller):
    """Flujo completo con prueba de vida y validación continua."""
    encodings, residentes = residente_manager.cargar_galeria()

    if not encodings:
        print("[ADVERTENCIA] No hay imágenes faciales válidas en la galería.")
        return

    print(f"[INFO] Codificaciones cargadas: {len(encodings)}")
    print("\nIniciando monitoreo en TIEMPO REAL.")
    print("Presione la tecla 'q' o 'ESC' en la ventana de video para finalizar.")

    prueba_vida = None
    esperando_retirada = False
    mensaje_resultado = None
    es_autorizado = False
    mostrar_resultado_hasta = 0.0

    with CameraManager() as camara:
        if not camara.esta_abierta():
            print("[ERROR] No se pudo abrir la cámara.")
            return

        while True:
            exito, frame = camara.leer_frame()
            if not exito or frame is None:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ubicaciones = detectar_rostros(frame_rgb)

            if esperando_retirada:
                # GUI de usuario: Muestra el resultado y espera que la persona salga del plano
                for ubicacion in ubicaciones:
                    if time.monotonic() < mostrar_resultado_hasta:
                        dibujar_interfaz_acceso(frame, ubicacion, mensaje_resultado, es_autorizado)
                    else:
                        dibujar_interfaz_acceso(frame, ubicacion, "Retirese para nuevo intento", False)

                if not ubicaciones:
                    esperando_retirada = False
                    prueba_vida = None
                    mensaje_resultado = None

            elif not ubicaciones:
                prueba_vida = None
                dibujar_texto_hud(frame, "Esperando una persona...", (255, 255, 255), 60)

            elif len(ubicaciones) > 1:
                prueba_vida = None
                for ubicacion in ubicaciones:
                    dibujar_interfaz_acceso(frame, ubicacion, "SOLO 1 PERSONA", False)
                dibujar_texto_hud(frame, "Debe haber una sola persona frente a la camara", (0, 0, 255), 60)

            else:
                if prueba_vida is None:
                    prueba_vida = PruebaVida()

                estado, caras_vida, ear, mar = prueba_vida.evaluar(frame)
                ubicacion = ubicaciones[0]

                # GUI de usuario en estado "Evaluando" (autorizado=None para color naranja)
                dibujar_interfaz_acceso(frame, ubicacion, estado, autorizado=None)

                # Instrucciones y métricas HUD (Lógica del compañero)
                dibujar_texto_hud(frame, prueba_vida.obtener_instruccion(), (0, 220, 255), 60)
                if ear is not None: dibujar_texto_hud(frame, f"EAR: {ear:.3f}", (255, 255, 255), 90)
                if mar is not None: dibujar_texto_hud(frame, f"MAR: {mar:.3f}", (255, 255, 255), 120)

                if prueba_vida.superada:
                    print("[OK] Prueba de vida superada.")
                    encoding_captura = codificar_rostro(frame_rgb)

                    if encoding_captura is None:
                        mensaje_resultado = "Error de lectura"
                        es_autorizado = False
                    else:
                        resultado = identificar_1aN(
                            encoding_captura, encodings, residentes, umbral=UMBRAL_DISTANCIA
                        )
                        autorizado, mensaje, evento = access_controller.procesar_identificacion(resultado)
                        es_autorizado = autorizado
                        mensaje_resultado = mensaje
                        if evento is not None:
                            print(f"[EVENTO] {evento}")

                    mostrar_resultado_hasta = time.monotonic() + 3.0
                    esperando_retirada = True

                elif prueba_vida.fallida:
                    mensaje_resultado = "Denegado: Prueba fallida"
                    es_autorizado = False
                    event_logger.registrar_acceso_denegado(motivo=mensaje_resultado)
                    mostrar_resultado_hasta = time.monotonic() + 3.0
                    esperando_retirada = True

            cv2.imshow("Terminal de Acceso - Tiempo Real", frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q") or tecla == 27:
                break

    cv2.destroyAllWindows()


def main():
    print("=" * 70)
    print("  SISTEMA DE ACCESO SIN LLAVES Y AUDITORÍA DE RESIDENTES (FUSIONADO)")
    print("=" * 70)

    # Inicialización de la lógica modular
    residente_manager = ResidenteManager()
    event_logger = EventLogger()
    access_controller = AccessController(
        event_logger=event_logger,
        tiempo_cooldown=TIEMPO_ENTRE_REGISTROS_SEG,
    )

    residentes = residente_manager.listar()
    print(f"[INFO] Residentes registrados en base: {len(residentes)}")
    for residente in residentes:
        estado = "AUTORIZADO" if residente.autorizado else "NO AUTORIZADO"
        print(f" - {residente.nombre}: {estado}")

    # Menú original mantenido
    print("\nSeleccione el modo de operación de la cámara del edificio:")
    print("  [1] Monitoreo continuo en Tiempo Real (con Prueba de Vida)")
    print("  [2] Captura única instantánea (Foto rápida sin liveness)")

    modo = input("\nIngrese opción (Enter para opción 1 por defecto): ").strip()

    if modo == "2":
        ejecutar_modo_captura_unica(residente_manager, event_logger, access_controller)
    else:
        ejecutar_modo_tiempo_real(residente_manager, event_logger, access_controller)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Programa interrumpido por el usuario.")
    except Exception as error:
        print(f"\n[ERROR DEL SISTEMA] {error}")