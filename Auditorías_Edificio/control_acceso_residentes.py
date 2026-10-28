"""
Script 2: Sistema de Acceso sin Llaves y Auditoría para Residentes (Tiempo Real / Cámara Web)
---------------------------------------------------------------------------------------------
Este módulo gestiona el control de acceso sin llaves para residentes de un edificio o
condominio mediante reconocimiento facial en tiempo real a través de una cámara web.

Funcionalidades principales:
1. Carga la base de datos de imágenes de residentes desde el directorio 'Residentes'.
2. Convierte cada imagen de BGR a RGB y codifica las características faciales.
3. Captura video/imágenes en tiempo real desde la cámara web.
4. Compara los rostros detectados con la base de datos y obtiene la menor distancia con numpy.
5. Si existe coincidencia dentro del umbral de seguridad, autoriza el acceso sin llave,
   muestra el recuadro con el nombre del residente y registra el evento de auditoría
   (Entrada / Salida con fecha y hora) en 'registro_accesos.csv'.
"""

import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

import cv2
import face_recognition as fr
import numpy


# Configuración de rutas y parámetros del sistema (sin rutas absolutas hardcodeadas)
DIRECTORIO_BASE = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_RESIDENTES = os.path.join(DIRECTORIO_BASE, "Residentes")
ARCHIVO_REGISTRO_CSV = os.path.join(DIRECTORIO_BASE, "registro_accesos.csv")

EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp")
UMBRAL_DISTANCIA_MAXIMA = 0.6
TIEMPO_ENTRE_REGISTROS_SEG = 15  # Cooldown en modo continuo para no duplicar el mismo cruce


def cargar_base_datos_residentes(ruta_directorio: str):
    """
    Recorre el directorio de residentes, carga cada imagen válida y extrae el nombre
    del residente a partir del nombre del archivo.

    Args:
        ruta_directorio (str): Ruta a la carpeta que contiene las fotos de los residentes.

    Returns:
        tuple: (lista_imagenes, nombres_residentes)

    Raises:
        FileNotFoundError: Si el directorio de residentes no existe.
        ValueError: Si no se encuentran imágenes válidas en el directorio.
    """
    if not os.path.isdir(ruta_directorio):
        raise FileNotFoundError(
            f"No se encontró el directorio de base de datos de residentes: {ruta_directorio}"
        )

    archivos = sorted(os.listdir(ruta_directorio))
    imagenes = []
    nombres_residentes = []

    for archivo in archivos:
        if not archivo.lower().endswith(EXTENSIONES_VALIDAS):
            continue

        ruta_completa = os.path.join(ruta_directorio, archivo)
        # En Windows, cv2.imread no soporta rutas con tildes o caracteres Unicode.
        # Se leen los bytes con numpy y se decodifican con cv2.imdecode para máxima compatibilidad.
        datos = numpy.fromfile(ruta_completa, dtype=numpy.uint8)
        imagen_actual = cv2.imdecode(datos, cv2.IMREAD_COLOR) if datos.size > 0 else None

        if imagen_actual is None:
            print(f"[ADVERTENCIA] No se pudo leer la imagen: {archivo}. Se omitirá.")
            continue

        imagenes.append(imagen_actual)
        nombre_residente = os.path.splitext(archivo)[0]
        nombres_residentes.append(nombre_residente)

    if not imagenes:
        raise ValueError(
            f"No se encontraron imágenes válidas en la carpeta '{ruta_directorio}'."
        )

    return imagenes, nombres_residentes


def codificar_residentes(imagenes, nombres_residentes):
    """
    Convierte la lista de imágenes de BGR a RGB y obtiene la codificación facial
    de cada residente registrado.

    Args:
        imagenes (list): Lista de imágenes cargadas con OpenCV (BGR).
        nombres_residentes (list): Lista de nombres correspondientes a cada imagen.

    Returns:
        tuple: (lista_codificada, nombres_validos) con las codificaciones y nombres
               de los residentes cuyo rostro fue detectado correctamente.
    """
    lista_codificada = []
    nombres_validos = []

    for imagen, nombre in zip(imagenes, nombres_residentes):
        # Convertir de BGR a RGB para su correcta interpretación por face_recognition
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        codificaciones = fr.face_encodings(imagen_rgb)

        if not codificaciones:
            print(
                f"[ADVERTENCIA] No se detectó un rostro claro en la foto de '{nombre}'. "
                "Se omitirá de la base activa."
            )
            continue

        lista_codificada.append(codificaciones[0])
        nombres_validos.append(nombre)

    return lista_codificada, nombres_validos


def inicializar_archivo_auditoria(ruta_csv: str):
    """
    Verifica si existe el archivo CSV de auditoría de accesos y, si no existe
    o está vacío, escribe la cabecera estándar.
    """
    if not os.path.exists(ruta_csv) or os.path.getsize(ruta_csv) == 0:
        with open(ruta_csv, mode="w", encoding="utf-8") as archivo:
            archivo.write("residente,fecha,hora,movimiento,estado\n")


def determinar_siguiente_movimiento(residente: str, ruta_csv: str) -> str:
    """
    Consulta el historial del archivo CSV para alternar automáticamente el tipo de
    movimiento del residente entre 'ENTRADA' y 'SALIDA' en la auditoría del edificio.

    Args:
        residente (str): Nombre del residente reconocido.
        ruta_csv (str): Ruta al archivo CSV de registro.

    Returns:
        str: 'ENTRADA' si no tiene registros previos o su último movimiento fue 'SALIDA';
             'SALIDA' si su último movimiento registrado fue 'ENTRADA'.
    """
    inicializar_archivo_auditoria(ruta_csv)

    ultimo_movimiento = None
    with open(ruta_csv, mode="r", encoding="utf-8") as archivo:
        lineas = archivo.readlines()

    for linea in lineas[1:]:  # Omitir cabecera
        partes = [campo.strip() for campo in linea.strip().split(",")]
        if len(partes) >= 4 and partes[0] == residente:
            ultimo_movimiento = partes[3]

    if ultimo_movimiento == "ENTRADA":
        return "SALIDA"
    return "ENTRADA"


def registrar_acceso_auditoria(residente: str, ruta_csv: str = ARCHIVO_REGISTRO_CSV) -> str:
    """
    Registra el ingreso o egreso del residente reconocido en el archivo CSV con
    fecha y hora exactas.

    Args:
        residente (str): Nombre del residente identificado.
        ruta_csv (str): Ruta al archivo CSV de auditoría.

    Returns:
        str: Tipo de movimiento registrado ('ENTRADA' o 'SALIDA').
    """
    inicializar_archivo_auditoria(ruta_csv)
    movimiento = determinar_siguiente_movimiento(residente, ruta_csv)

    ahora = datetime.now()
    str_fecha = ahora.strftime("%Y-%m-%d")
    str_hora = ahora.strftime("%H:%M:%S")

    with open(ruta_csv, mode="a", encoding="utf-8") as archivo:
        archivo.write(f"{residente}, {str_fecha}, {str_hora}, {movimiento}, ACCESO SIN LLAVE\n")

    print(
        f"[AUDITORÍA] {str_fecha} {str_hora} | Residente: {residente} | "
        f"Movimiento: {movimiento} | Estado: ACCESO SIN LLAVE AUTORIZADO"
    )
    return movimiento


def dibujar_interfaz_acceso(imagen, ubicacion_cara, texto_etiqueta: str, autorizado: bool):
    """
    Dibuja el rectángulo delimitador alrededor del rostro detectado y el cartel
    con el nombre del residente (o alerta de desconocido) sobre la imagen capturada.
    """
    y1, x2, y2, x1 = ubicacion_cara
    color = (0, 200, 0) if autorizado else (0, 0, 220)

    # Rectángulo principal alrededor del rostro
    cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 2)
    # Rectángulo relleno inferior para el texto
    cv2.rectangle(imagen, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
    # Texto con el nombre del residente y estado
    cv2.putText(
        imagen,
        texto_etiqueta,
        (x1 + 6, y2 - 8),
        cv2.FONT_HERSHEY_COMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


def abrir_camara(indice_camara: int = 0):
    """
    Intenta abrir la cámara web en Windows (usando CAP_DSHOW) con fallback automático
    al backend por defecto si es necesario.
    """
    captura = cv2.VideoCapture(indice_camara, cv2.CAP_DSHOW)
    if not captura.isOpened():
        captura = cv2.VideoCapture(indice_camara)
    return captura


def procesar_fotograma(
    imagen,
    lista_residentes_codificada,
    nombres_residentes,
    ultimos_registros: dict
):
    """
    Detecta y codifica los rostros presentes en un fotograma, los compara contra la base
    de datos de residentes y actualiza tanto la imagen como el registro de auditoría.

    Returns:
        bool: True si se detectó al menos un rostro en el fotograma, False en caso contrario.
    """
    # Convertir fotograma de BGR a RGB para face_recognition
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Localizar y codificar caras en la imagen capturada
    caras_captura = fr.face_locations(imagen_rgb)
    caras_captura_codificadas = fr.face_encodings(imagen_rgb, caras_captura)

    for cara_codif, cara_ubic in zip(caras_captura_codificadas, caras_captura):
        coincidencias = fr.compare_faces(
            lista_residentes_codificada, cara_codif, tolerance=UMBRAL_DISTANCIA_MAXIMA
        )
        distancias = fr.face_distance(lista_residentes_codificada, cara_codif)

        print(f"Distancias calculadas contra la base de residentes: {distancias.round(3)}")

        indice_coincidencia = int(numpy.argmin(distancias))
        distancia_minima = float(distancias[indice_coincidencia])

        if distancia_minima > UMBRAL_DISTANCIA_MAXIMA or not coincidencias[indice_coincidencia]:
            print("[ALERTA] Rostro no reconocido: No coincide con ningún residente registrado.")
            dibujar_interfaz_acceso(
                imagen, cara_ubic, "ACCESO DENEGADO", autorizado=False
            )
        else:
            nombre = nombres_residentes[indice_coincidencia]
            ahora_ts = datetime.now().timestamp()
            ultimo_ts = ultimos_registros.get(nombre, 0.0)

            # Evitar múltiples registros seguidos en el mismo segundo durante video en vivo
            if (ahora_ts - ultimo_ts) >= TIEMPO_ENTRE_REGISTROS_SEG:
                movimiento = registrar_acceso_auditoria(nombre)
                ultimos_registros[nombre] = ahora_ts
            else:
                # Si ya se registró hace pocos segundos, mantener visible su estado actual
                movimiento = "AUTORIZADO"

            etiqueta = f"{nombre} ({movimiento})"
            dibujar_interfaz_acceso(imagen, cara_ubic, etiqueta, autorizado=True)

    return len(caras_captura) > 0


def ejecutar_modo_captura_unica(lista_residentes_codificada, nombres_residentes):
    """
    Toma una captura instantánea desde la cámara web, verifica la identidad del residente,
    registra el acceso en el CSV y muestra la imagen en pantalla.
    """
    print("\nIniciando cámara web para captura de acceso...")
    captura = abrir_camara(0)
    exito, imagen = captura.read()
    captura.release()

    if not exito or imagen is None:
        print("[ERROR] No se ha podido tomar la captura desde la cámara web.")
        return

    ultimos_registros = {}
    hubo_rostros = procesar_fotograma(
        imagen, lista_residentes_codificada, nombres_residentes, ultimos_registros
    )

    if not hubo_rostros:
        print("[INFO] No se detectaron rostros frente a la cámara en la captura.")
        cv2.putText(
            imagen,
            "Sin rostros detectados",
            (30, 45),
            cv2.FONT_HERSHEY_COMPLEX,
            0.9,
            (0, 165, 255),
            2
        )

    cv2.imshow("Terminal de Acceso sin Llaves - Captura Web", imagen)
    print("Presione cualquier tecla en la ventana de imagen para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ejecutar_modo_tiempo_real(lista_residentes_codificada, nombres_residentes):
    """
    Mantiene la cámara web activa en tiempo real (streaming de video), reconociendo a los
    residentes que se acercan a la entrada del edificio y registrando su entrada/salida.
    """
    print("\nIniciando monitoreo en TIEMPO REAL en acceso del edificio...")
    print("Presione la tecla 'q' o 'ESC' en la ventana de video para finalizar.")

    captura = abrir_camara(0)
    if not captura.isOpened():
        print("[ERROR] No se pudo abrir la cámara web.")
        return

    ultimos_registros = {}

    try:
        while True:
            exito, fotograma = captura.read()
            if not exito or fotograma is None:
                print("[ERROR] No se pudo leer el fotograma de la cámara web.")
                break

            procesar_fotograma(
                fotograma, lista_residentes_codificada, nombres_residentes, ultimos_registros
            )

            cv2.putText(
                fotograma,
                "Acceso Condominio - Presione 'q' para salir",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.imshow("Terminal de Acceso sin Llaves - Tiempo Real", fotograma)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q") or tecla == 27:  # 'q' o ESC
                break
    finally:
        captura.release()
        cv2.destroyAllWindows()


def main():
    print("=" * 70)
    print("  SISTEMA DE ACCESO SIN LLAVES Y AUDITORÍA DE RESIDENTES (EDIFICIO)")
    print("=" * 70)

    # 1. Cargar base de datos de residentes
    print(f"Cargando base de datos de residentes desde: '{os.path.basename(DIRECTORIO_RESIDENTES)}'...")
    mis_imagenes, nombres_residentes = cargar_base_datos_residentes(DIRECTORIO_RESIDENTES)
    print(f"Residentes encontrados ({len(nombres_residentes)}): {nombres_residentes}")

    # 2. Codificar imágenes de los residentes
    print("Codificando características biométricas de los residentes...")
    lista_residentes_codificada, nombres_residentes_validos = codificar_residentes(
        mis_imagenes, nombres_residentes
    )
    print(f"Base biométrica lista con {len(nombres_residentes_validos)} residentes activos.")

    # 3. Inicializar archivo de registro de auditoría CSV
    inicializar_archivo_auditoria(ARCHIVO_REGISTRO_CSV)

    # 4. Seleccionar modo de operación de cámara web
    print("\nSeleccione el modo de operación de la cámara del edificio:")
    print("  [1] Monitoreo continuo en Tiempo Real (video en vivo - presionar 'q' para salir)")
    print("  [2] Captura única instantánea (toma una foto, verifica y registra acceso)")

    modo = input("\nIngrese opción (Enter para opción 1 por defecto): ").strip()

    if modo == "2":
        ejecutar_modo_captura_unica(lista_residentes_codificada, nombres_residentes_validos)
    else:
        ejecutar_modo_tiempo_real(lista_residentes_codificada, nombres_residentes_validos)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n[ERROR DEL SISTEMA] {error}")
