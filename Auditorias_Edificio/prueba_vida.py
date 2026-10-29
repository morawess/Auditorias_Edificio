"""Prueba independiente de los desafios aleatorios de vida."""

import cv2

from camera import CameraManager
from liveness import PruebaVida


def ejecutar_prueba_vida():
    prueba = PruebaVida()

    with CameraManager() as camara:
        if not camara.esta_abierta():
            print("[ERROR] No se pudo abrir la camara.")
            return False

        resultado = None

        while True:
            lectura_correcta, frame = camara.leer_frame()

            if not lectura_correcta or frame is None:
                print("[ERROR] No se pudo leer la camara.")
                break

            estado, ubicaciones, ear, mar = prueba.evaluar(frame)

            color = (0, 220, 255)

            if prueba.superada:
                resultado = True
                estado = "PRUEBA DE VIDA SUPERADA"
                color = (0, 200, 0)

            elif prueba.fallida:
                resultado = False
                estado = "PRUEBA DE VIDA FALLIDA"
                color = (0, 0, 255)

            for arriba, derecha, abajo, izquierda in ubicaciones:
                cv2.rectangle(
                    frame,
                    (izquierda, arriba),
                    (derecha, abajo),
                    color,
                    2,
                )

            cv2.putText(
                frame,
                prueba.obtener_instruccion(),
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

            cv2.putText(
                frame,
                estado,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

            if ear is not None:
                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )

            if mar is not None:
                cv2.putText(
                    frame,
                    f"MAR: {mar:.3f}",
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )

            cv2.putText(
                frame,
                "Presiona Q para salir",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Prueba de vida - ResQ Edificio", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if resultado is not None:
                # Mantiene el resultado visible hasta presionar Q.
                continue

    cv2.destroyAllWindows()

    if resultado is True:
        print("[OK] Prueba de vida superada.")
    elif resultado is False:
        print("[RECHAZADO] No se completo el desafio a tiempo.")
    else:
        print("[CANCELADO] Prueba interrumpida.")

    return resultado is True


if __name__ == "__main__":
    ejecutar_prueba_vida()