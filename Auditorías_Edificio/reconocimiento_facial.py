"""Verificación facial 1:1 entre dos imágenes estáticas.

Este script es el punto de entrada del Sprint 2. Reutiliza las funciones de
`face_engine.py` y no contiene la implementación del algoritmo facial.

Uso desde la carpeta del proyecto:
    python reconocimiento_facial.py
    python reconocimiento_facial.py FotoA.jpg FotoB.jpg
    python reconocimiento_facial.py FotoA.jpg FotoC.jpg
    python reconocimiento_facial.py FotoA.jpg FotoB.jpg --sin-ventanas

Las imágenes deben contener exactamente un rostro detectable cada una.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import face_engine as motor
from config import UMBRAL_DISTANCIA
from models import ResultadoVerificacion


DIRECTORIO_PROYECTO = Path(__file__).resolve().parent
UbicacionRostro = Tuple[int, int, int, int]


def resolver_ruta(ruta: str | Path) -> Path:
    """Resuelve una ruta absoluta o relativa de forma predecible.

    Para rutas relativas, primero busca desde el directorio actual de la
    terminal. Si no existe allí, prueba desde el directorio del proyecto.
    """
    candidata = Path(ruta).expanduser()

    if candidata.is_absolute():
        return candidata.resolve()

    desde_terminal = (Path.cwd() / candidata).resolve()
    if desde_terminal.exists():
        return desde_terminal

    return (DIRECTORIO_PROYECTO / candidata).resolve()


def _cargar_un_rostro(
    ruta: Path,
    etiqueta: str,
) -> tuple[object, UbicacionRostro, object]:
    """Carga una imagen y extrae la codificación de su único rostro.

    `face_recognition.load_image_file`, usado por `motor.cargar_imagen_rgb`,
    ya entrega la imagen en RGB. No se debe volver a convertir BGR a RGB.
    """
    es_valida, mensaje = motor.validar_archivo_imagen(str(ruta))
    if not es_valida:
        raise ValueError(f"{etiqueta}: {mensaje}")

    try:
        imagen_rgb = motor.cargar_imagen_rgb(str(ruta))
    except (OSError, ValueError, RuntimeError) as error:
        raise ValueError(f"{etiqueta}: no se pudo leer la imagen. {error}") from error

    ubicaciones = motor.detectar_rostros(imagen_rgb)

    if len(ubicaciones) == 0:
        raise ValueError(
            f"{etiqueta}: no se detectó ningún rostro. "
            "Utilizá una imagen nítida en la que el rostro sea visible."
        )

    if len(ubicaciones) > 1:
        raise ValueError(
            f"{etiqueta}: se detectaron {len(ubicaciones)} rostros. "
            "Para la comparación 1:1, cada imagen debe contener exactamente uno."
        )

    ubicacion = ubicaciones[0]
    encoding = motor.codificar_rostro(imagen_rgb, ubicacion)
    if encoding is None:
        raise ValueError(
            f"{etiqueta}: se detectó el rostro, pero no se pudieron codificar "
            "sus características faciales. Probá con otra imagen."
        )

    return imagen_rgb, ubicacion, encoding


def _crear_imagen_visual(
    imagen_rgb: object,
    ubicacion: UbicacionRostro,
    etiqueta: str,
    coincide: bool,
):
    """Convierte RGB a BGR únicamente para dibujar y mostrar con OpenCV."""
    import cv2

    imagen_bgr = motor.rgb_a_bgr(imagen_rgb.copy())
    arriba, derecha, abajo, izquierda = ubicacion
    color = (0, 170, 0) if coincide else (0, 0, 220)

    cv2.rectangle(
        imagen_bgr,
        (izquierda, arriba),
        (derecha, abajo),
        color,
        3,
    )
    cv2.putText(
        imagen_bgr,
        etiqueta,
        (max(izquierda, 8), max(arriba - 12, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )
    return imagen_bgr


def _mostrar_imagenes(
    imagen_referencia: object,
    ubicacion_referencia: UbicacionRostro,
    imagen_prueba: object,
    ubicacion_prueba: UbicacionRostro,
    coincide: bool,
) -> None:
    """Muestra ambas imágenes. Un fallo de GUI no invalida la comparación."""
    cv2 = None
    resultado = "COINCIDENCIA" if coincide else "SIN COINCIDENCIA"

    try:
        import cv2 as cv2_modulo
        cv2 = cv2_modulo

        referencia_bgr = _crear_imagen_visual(
            imagen_referencia,
            ubicacion_referencia,
            f"REFERENCIA - {resultado}",
            coincide,
        )
        prueba_bgr = _crear_imagen_visual(
            imagen_prueba,
            ubicacion_prueba,
            f"PRUEBA - {resultado}",
            coincide,
        )

        cv2.namedWindow("Imagen de referencia", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Imagen de prueba", cv2.WINDOW_NORMAL)
        cv2.imshow("Imagen de referencia", referencia_bgr)
        cv2.imshow("Imagen de prueba", prueba_bgr)
        print("\nSe abrieron las ventanas de las dos imágenes.")
        print("Presioná cualquier tecla con una ventana seleccionada para cerrarlas.")
        cv2.waitKey(0)

    except Exception as error:
        print(
            "[ADVERTENCIA] La comparación se realizó, pero no se pudieron "
            f"mostrar las ventanas de OpenCV: {error}"
        )
        print("Podés volver a ejecutar el comando con --sin-ventanas.")
    finally:
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


def comparar_imagenes(
    ruta_referencia: str | Path,
    ruta_prueba: str | Path,
    umbral: float = UMBRAL_DISTANCIA,
    mostrar: bool = False,
) -> ResultadoVerificacion:
    """Compara los rostros únicos de dos imágenes y devuelve el resultado.

    Esta función puede ser reutilizada por pruebas automatizadas o por otros
    módulos. No autoriza accesos ni registra entradas: esas responsabilidades
    pertenecen a otros componentes del sistema.
    """
    if not isinstance(umbral, (int, float)) or not math.isfinite(float(umbral)):
        raise ValueError("El umbral debe ser un número finito.")
    if float(umbral) < 0:
        raise ValueError("El umbral no puede ser negativo.")

    ruta_ref = resolver_ruta(ruta_referencia)
    ruta_test = resolver_ruta(ruta_prueba)

    imagen_ref, ubicacion_ref, encoding_ref = _cargar_un_rostro(
        ruta_ref,
        "Imagen de referencia",
    )
    imagen_test, ubicacion_test, encoding_test = _cargar_un_rostro(
        ruta_test,
        "Imagen de prueba",
    )

    resultado = motor.verificar_1a1(
        encoding_ref,
        encoding_test,
        umbral=float(umbral),
    )

    if mostrar:
        _mostrar_imagenes(
            imagen_ref,
            ubicacion_ref,
            imagen_test,
            ubicacion_test,
            resultado.coincide,
        )

    return resultado


def _umbral_argumento(valor: str) -> float:
    """Valida el argumento --umbral de argparse."""
    try:
        umbral = float(valor)
    except ValueError as error:
        raise argparse.ArgumentTypeError("el umbral debe ser numérico") from error

    if not math.isfinite(umbral) or umbral < 0:
        raise argparse.ArgumentTypeError(
            "el umbral debe ser un número finito mayor o igual que cero"
        )
    return umbral


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compara un rostro de referencia con un rostro de prueba. "
            "Cada imagen debe contener exactamente un rostro detectable."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "imagen_referencia",
        nargs="?",
        default="FotoA.jpg",
        help="ruta de la imagen de referencia",
    )
    parser.add_argument(
        "imagen_prueba",
        nargs="?",
        default="FotoB.jpg",
        help="ruta de la imagen que se desea verificar",
    )
    parser.add_argument(
        "--umbral",
        type=_umbral_argumento,
        default=UMBRAL_DISTANCIA,
        help=(
            "distancia máxima para considerar una coincidencia; "
            "no representa un porcentaje de certeza"
        ),
    )
    parser.add_argument(
        "--sin-ventanas",
        action="store_true",
        help="no abrir ventanas gráficas de OpenCV",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="mostrar la traza completa si ocurre un error inesperado",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = crear_parser()
    args = parser.parse_args(argv)

    ruta_referencia = resolver_ruta(args.imagen_referencia)
    ruta_prueba = resolver_ruta(args.imagen_prueba)

    print("=" * 68)
    print(" VERIFICACIÓN FACIAL 1:1 — SPRINT 2")
    print("=" * 68)
    print(f"Referencia: {ruta_referencia}")
    print(f"Imagen de prueba: {ruta_prueba}")
    print(f"Umbral configurado: {float(args.umbral):.4f}")

    try:
        resultado = comparar_imagenes(
            ruta_referencia,
            ruta_prueba,
            umbral=args.umbral,
            mostrar=not args.sin_ventanas,
        )
    except Exception as error:
        if args.debug:
            raise
        print(f"\n[ERROR] {error}")
        print(
            "Verificá que las rutas existan, que las imágenes sean válidas "
            "y que las dependencias estén instaladas en el entorno virtual activo."
        )
        return 2

    print("\n" + "-" * 68)
    if resultado.coincide:
        print("RESULTADO: COINCIDENCIA")
    else:
        print("RESULTADO: SIN COINCIDENCIA")
    print(f"Distancia facial: {resultado.distancia:.4f}")
    print(f"Umbral: {float(args.umbral):.4f}")
    print("La distancia no es un porcentaje ni una probabilidad de identidad.")
    print("Esta prueba no concede ni deniega por sí sola el acceso al edificio.")
    print("-" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
