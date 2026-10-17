"""
Script 1: Verificación Biométrica de Identidad para Residentes (Comparación de Imágenes)
----------------------------------------------------------------------------------------
Este módulo carga y procesa dos imágenes (una imagen de control de la base de datos del
edificio/condominio y una imagen de prueba), localiza los rostros presentes, codifica
sus características biométricas y los compara para verificar si pertenecen al mismo
residente. Muestra ambas imágenes en pantalla con el recuadro del rostro detectado,
indicando si el acceso es concedido (coincidencia) o denegado junto con la distancia
biométrica calculada.
"""

import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

import cv2
import face_recognition as fr


# Configuraciones por defecto del sistema de verificación
DIRECTORIO_BASE = os.path.dirname(os.path.abspath(__file__))
IMAGEN_CONTROL_DEFAULT = os.path.join(DIRECTORIO_BASE, "FotoA.jpg")
IMAGEN_PRUEBA_DEFAULT = os.path.join(DIRECTORIO_BASE, "FotoB.jpg")
TOLERANCIA_POR_DEFECTO = 0.6


def cargar_y_procesar_imagen(ruta_imagen: str):
    """
    Carga una imagen desde un archivo y la convierte de RGB (formato de face_recognition)
    a BGR (formato utilizado por OpenCV para visualización).

    Args:
        ruta_imagen (str): Ruta al archivo de imagen.

    Returns:
        numpy.ndarray: Imagen procesada en formato BGR lista para OpenCV y face_recognition.

    Raises:
        FileNotFoundError: Si el archivo de imagen no existe en la ruta indicada.
    """
    if not os.path.isfile(ruta_imagen):
        raise FileNotFoundError(f"No se encontró el archivo de imagen en: {ruta_imagen}")

    imagen_cargada = fr.load_image_file(ruta_imagen)
    # Convertir la imagen de BGR a RGB / viceversa para su correcta interpretación en OpenCV
    imagen_procesada = cv2.cvtColor(imagen_cargada, cv2.COLOR_BGR2RGB)
    return imagen_procesada


def detectar_y_codificar_rostro(imagen, nombre_referencia: str = "Imagen"):
    """
    Localiza el rostro principal en la imagen y codifica sus características faciales.

    Args:
        imagen (numpy.ndarray): Imagen cargada y procesada.
        nombre_referencia (str): Etiqueta descriptiva para mensajes de error.

    Returns:
        tuple: (ubicacion_rostro, codificacion_rostro) donde ubicacion_rostro es
               (top, right, bottom, left) y codificacion_rostro es el vector de 128 dimensiones.

    Raises:
        ValueError: Si no se detecta ningún rostro en la imagen proporcionada.
    """
    ubicaciones = fr.face_locations(imagen)
    if not ubicaciones:
        raise ValueError(f"No se pudo detectar ningún rostro en '{nombre_referencia}'.")

    ubicacion_principal = ubicaciones[0]
    codificaciones = fr.face_encodings(imagen, [ubicacion_principal])
    if not codificaciones:
        raise ValueError(f"No se pudieron codificar las características faciales en '{nombre_referencia}'.")

    return ubicacion_principal, codificaciones[0]


def dibujar_recuadro_rostro(imagen, ubicacion_rostro, color=(0, 255, 0), grosor: int = 2):
    """
    Dibuja un rectángulo alrededor del rostro detectado en la imagen.

    Args:
        imagen (numpy.ndarray): Imagen sobre la cual dibujar.
        ubicacion_rostro (tuple): Coordenadas (top, right, bottom, left) del rostro.
        color (tuple): Color BGR para el rectángulo.
        grosor (int): Grosor de las líneas del rectángulo.
    """
    arriba, derecha, abajo, izquierda = ubicacion_rostro
    cv2.rectangle(
        imagen,
        (izquierda, arriba),
        (derecha, abajo),
        color,
        grosor
    )


def comparar_rostros_residente(
    ruta_control: str = IMAGEN_CONTROL_DEFAULT,
    ruta_prueba: str = IMAGEN_PRUEBA_DEFAULT,
    tolerancia: float = TOLERANCIA_POR_DEFECTO
):
    """
    Ejecuta el flujo completo de comparación entre la imagen de control del residente
    y la imagen de prueba capturada, mostrando los resultados visuales en pantalla.
    """
    print("=" * 65)
    print("  SISTEMA DE ACCESO SIN LLAVES - VERIFICACIÓN DE IDENTIDAD")
    print("=" * 65)
    print(f"Cargando foto de control (Residente registrado): {os.path.basename(ruta_control)}")
    print(f"Cargando foto de prueba  (Intento de acceso)   : {os.path.basename(ruta_prueba)}")

    # 1. Cargar y convertir imágenes para su correcta interpretación
    foto_control = cargar_y_procesar_imagen(ruta_control)
    foto_prueba = cargar_y_procesar_imagen(ruta_prueba)

    # 2. Localizar rostros y codificar características faciales
    lugar_cara_control, cara_codificada_control = detectar_y_codificar_rostro(
        foto_control, "Foto Control"
    )
    lugar_cara_prueba, cara_codificada_prueba = detectar_y_codificar_rostro(
        foto_prueba, "Foto Prueba"
    )

    # 3. Comparación de imágenes y cálculo de distancia facial
    resultado = fr.compare_faces(
        [cara_codificada_control],
        cara_codificada_prueba,
        tolerance=tolerancia
    )
    distancia = fr.face_distance([cara_codificada_control], cara_codificada_prueba)

    coincide = bool(resultado[0])
    valor_distancia = float(distancia[0])

    # Imprimir resultados por consola tal como lo requiere la consigna
    print("-" * 65)
    print(f"Resultado de coincidencia (compare_faces): {resultado}")
    print(f"Distancia facial calculada (face_distance): {distancia}")
    estado_acceso = "ACCESO AUTORIZADO (Coincide)" if coincide else "ACCESO DENEGADO (No coincide)"
    print(f"Estado de verificación                   : {estado_acceso}")
    print("=" * 65)

    # 4. Visualización gráfica: dibujar rectángulos y mostrar resultado sobre la imagen
    color_resultado = (0, 255, 0) if coincide else (0, 0, 255)
    dibujar_recuadro_rostro(foto_control, lugar_cara_control, color=(0, 255, 0))
    dibujar_recuadro_rostro(foto_prueba, lugar_cara_prueba, color=color_resultado)

    texto_pantalla = f"{resultado} {distancia.round(2)}"
    etiqueta_acceso = "ACCESO PERMITIDO" if coincide else "ACCESO DENEGADO"

    cv2.putText(
        foto_prueba,
        texto_pantalla,
        (30, 45),
        cv2.FONT_HERSHEY_COMPLEX,
        0.9,
        color_resultado,
        2
    )
    cv2.putText(
        foto_prueba,
        etiqueta_acceso,
        (30, 85),
        cv2.FONT_HERSHEY_COMPLEX,
        0.75,
        color_resultado,
        2
    )

    # 5. Mostrar las imágenes de control y prueba en pantalla
    cv2.imshow("Foto Control - Residente Registrado", foto_control)
    cv2.imshow("Foto Prueba - Verificacion de Acceso", foto_prueba)

    print("Presione cualquier tecla en la ventana de imagen para cerrar el programa...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def seleccionar_imagenes_interactivo():
    """
    Permite al usuario elegir con qué imagen de prueba comparar la foto de control
    o ingresar rutas personalizadas sin necesidad de modificar el código fuente.
    """
    imagenes_disponibles = [
        f for f in os.listdir(DIRECTORIO_BASE)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print("\nImágenes disponibles en el directorio del proyecto:")
    for idx, nombre_img in enumerate(imagenes_disponibles, start=1):
        print(f"  [{idx}] {nombre_img}")

    print("\nOpciones de ejecución:")
    print("  [1] Comparar FotoA.jpg (Control) vs FotoB.jpg (Prueba - Misma persona)")
    print("  [2] Comparar FotoA.jpg (Control) vs FotoC.jpg (Prueba - Otra persona)")
    print("  [3] Seleccionar imágenes manualmente de la lista")

    opcion = input("\nSeleccione una opción (Enter para opción 1 por defecto): ").strip()

    if opcion == "2":
        ruta_prueba = os.path.join(DIRECTORIO_BASE, "FotoC.jpg")
        return IMAGEN_CONTROL_DEFAULT, ruta_prueba
    elif opcion == "3" and imagenes_disponibles:
        try:
            idx_ctrl = int(input("Ingrese el número de la Foto de Control: ").strip()) - 1
            idx_prb = int(input("Ingrese el número de la Foto de Prueba: ").strip()) - 1
            ruta_ctrl = os.path.join(DIRECTORIO_BASE, imagenes_disponibles[idx_ctrl])
            ruta_prb = os.path.join(DIRECTORIO_BASE, imagenes_disponibles[idx_prb])
            return ruta_ctrl, ruta_prb
        except (ValueError, IndexError):
            print("Selección no válida. Se utilizarán FotoA.jpg y FotoB.jpg por defecto.")

    return IMAGEN_CONTROL_DEFAULT, IMAGEN_PRUEBA_DEFAULT


if __name__ == "__main__":
    try:
        img_control, img_prueba = seleccionar_imagenes_interactivo()
        comparar_rostros_residente(img_control, img_prueba)
    except Exception as error:
        print(f"\n[ERROR] No se pudo completar la comparación: {error}")
