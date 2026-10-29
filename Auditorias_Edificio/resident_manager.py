"""
Gestión de residentes del edificio.

Maneja el registro, consulta, actualización y eliminación de residentes,
así como la asociación de imágenes faciales de referencia. Los datos se
persisten en un archivo JSON independiente de los nombres de archivo.

La identidad del residente se basa en su UUID, no en el nombre del archivo
de imagen. Esto permite renombrar archivos, asociar múltiples imágenes
a un mismo residente y evitar duplicados accidentales.
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    ARCHIVO_RESIDENTES_DB,
    DIRECTORIO_DATOS,
    DIRECTORIO_RESIDENTES,
    EXTENSIONES_IMAGEN_VALIDAS,
)
from models import Residente


class ResidenteManager:
    """
    Administrador de la base de datos de residentes.

    Gestiona el ciclo de vida completo de los registros de residentes,
    incluyendo la importación inicial desde el directorio de imágenes
    existente y la persistencia en JSON.
    """

    def __init__(
        self,
        ruta_db: str = ARCHIVO_RESIDENTES_DB,
        ruta_imagenes: str = DIRECTORIO_RESIDENTES,
        ruta_datos: str = DIRECTORIO_DATOS,
    ):
        self.ruta_db = ruta_db
        self.ruta_imagenes = ruta_imagenes
        self.ruta_datos = ruta_datos
        self._residentes: Dict[str, Residente] = {}
        self._inicializar()

    def _inicializar(self):
        """Crea directorios necesarios y carga la base de datos."""
        os.makedirs(self.ruta_datos, exist_ok=True)
        os.makedirs(self.ruta_imagenes, exist_ok=True)

        if os.path.isfile(self.ruta_db):
            self._cargar_db()
        else:
            self._residentes = {}
            self._importar_directorio_existente()
            self._guardar_db()

    def _cargar_db(self):
        """Carga la base de datos de residentes desde el archivo JSON."""
        try:
            with open(self.ruta_db, "r", encoding="utf-8") as f:
                datos = json.load(f)

            self._residentes = {}
            for rid, rdata in datos.get("residentes", {}).items():
                self._residentes[rid] = Residente.from_dict(rdata)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[ERROR] No se pudo leer la base de datos de residentes: {e}")
            print("[INFO] Se creará una base nueva preservando el archivo anterior.")
            backup = self.ruta_db + ".bak"
            if os.path.exists(self.ruta_db):
                shutil.copy2(self.ruta_db, backup)
            self._residentes = {}
            self._importar_directorio_existente()
            self._guardar_db()

    def _guardar_db(self):
        """Persiste la base de datos de residentes al archivo JSON."""
        datos = {
            "residentes": {
                rid: res.to_dict() for rid, res in self._residentes.items()
            },
            "ultima_actualizacion": datetime.now().astimezone().isoformat(),
        }
        try:
            with open(self.ruta_db, "w", encoding="utf-8") as f:
                json.dump(datos, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"[ERROR] No se pudo guardar la base de datos: {e}")

    def _importar_directorio_existente(self):
        """
        Importa residentes desde el directorio de imágenes existente.

        Esta función se ejecuta una vez, al crear la base de datos por primera vez.
        Cada imagen válida en el directorio de residentes genera un registro nuevo
        con UUID propio. El nombre del residente se extrae del nombre del archivo.
        """
        if not os.path.isdir(self.ruta_imagenes):
            return

        archivos = sorted(os.listdir(self.ruta_imagenes))
        importados = 0

        for archivo in archivos:
            if not archivo.lower().endswith(EXTENSIONES_IMAGEN_VALIDAS):
                continue

            nombre = os.path.splitext(archivo)[0]

            # Evitar duplicados: buscar si ya existe un residente con este nombre
            existente = self.buscar_por_nombre(nombre)
            if existente:
                # Agregar imagen adicional si no está ya asociada
                if archivo not in existente.imagenes:
                    existente.imagenes.append(archivo)
                continue

            residente = Residente(
                nombre=nombre,
                autorizado=True,
                imagenes=[archivo],
            )
            self._residentes[residente.id] = residente
            importados += 1

        if importados > 0:
            print(f"[INFO] Se importaron {importados} residentes desde '{os.path.basename(self.ruta_imagenes)}'.")

    def registrar_residente(
        self,
        nombre: str,
        ruta_imagen: str,
        autorizado: bool = True,
        notas: str = "",
    ) -> Tuple[Optional[Residente], str]:
        """
        Registra un nuevo residente con su imagen de referencia.
        """
        if not nombre or not nombre.strip():
            return None, "El nombre del residente no puede estar vacío."

        nombre = nombre.strip()

        from face_engine import validar_archivo_imagen, cargar_imagen_rgb, detectar_rostros
        valida, msg = validar_archivo_imagen(ruta_imagen)
        if not valida:
            return None, f"Imagen no válida: {msg}"

        try:
            imagen_rgb = cargar_imagen_rgb(ruta_imagen)
            ubicaciones = detectar_rostros(imagen_rgb)
        except Exception as e:
            return None, f"Error al procesar la imagen: {e}"

        if not ubicaciones:
            return None, (
                "No se detectó ningún rostro en la imagen proporcionada. "
                "La imagen debe contener un rostro claramente visible."
            )

        if len(ubicaciones) > 1:
            return None, (
                f"Se detectaron {len(ubicaciones)} rostros en la imagen. "
                f"La imagen de referencia debe contener un único rostro."
            )

        ext = os.path.splitext(ruta_imagen)[1]
        nombre_archivo = self._generar_nombre_archivo_seguro(nombre, ext)
        ruta_destino = os.path.join(self.ruta_imagenes, nombre_archivo)

        ruta_origen_abs = os.path.abspath(ruta_imagen)
        ruta_destino_abs = os.path.abspath(ruta_destino)
        if ruta_origen_abs != ruta_destino_abs:
            try:
                shutil.copy2(ruta_imagen, ruta_destino)
            except OSError as e:
                return None, f"No se pudo copiar la imagen al directorio de residentes: {e}"

        existente = self.buscar_por_nombre(nombre)
        if existente:
            if nombre_archivo not in existente.imagenes:
                existente.imagenes.append(nombre_archivo)
            self._guardar_db()
            return existente, (
                f"El residente '{nombre}' ya existía. Se agregó una imagen "
                f"adicional de referencia ({nombre_archivo})."
            )

        residente = Residente(
            nombre=nombre,
            autorizado=autorizado,
            imagenes=[nombre_archivo],
            notas=notas,
        )
        self._residentes[residente.id] = residente
        self._guardar_db()

        return residente, f"Residente '{nombre}' registrado exitosamente (ID: {residente.id[:8]}...)."

    def agregar_imagen(
        self, residente_id: str, ruta_imagen: str
    ) -> Tuple[bool, str]:
        """
        Agrega una imagen de referencia adicional a un residente existente.
        """
        residente = self.obtener(residente_id)
        if not residente:
            return False, f"No se encontró un residente con ID '{residente_id[:8]}...'."

        from face_engine import validar_archivo_imagen, cargar_imagen_rgb, detectar_rostros

        valida, msg = validar_archivo_imagen(ruta_imagen)
        if not valida:
            return False, f"Imagen no válida: {msg}"

        try:
            imagen_rgb = cargar_imagen_rgb(ruta_imagen)
            ubicaciones = detectar_rostros(imagen_rgb)
        except Exception as e:
            return False, f"Error al procesar la imagen: {e}"

        if not ubicaciones:
            return False, "No se detectó ningún rostro en la imagen."
        if len(ubicaciones) > 1:
            return False, f"Se detectaron {len(ubicaciones)} rostros. Se requiere un único rostro."

        ext = os.path.splitext(ruta_imagen)[1]
        sufijo = len(residente.imagenes) + 1
        nombre_archivo = self._generar_nombre_archivo_seguro(
            f"{residente.nombre}_{sufijo}", ext
        )
        ruta_destino = os.path.join(self.ruta_imagenes, nombre_archivo)

        try:
            shutil.copy2(ruta_imagen, ruta_destino)
        except OSError as e:
            return False, f"No se pudo copiar la imagen: {e}"

        residente.imagenes.append(nombre_archivo)
        self._guardar_db()

        return True, f"Imagen '{nombre_archivo}' agregada al residente '{residente.nombre}'."

    def obtener(self, residente_id: str) -> Optional[Residente]:
        """Obtiene un residente por su UUID."""
        return self._residentes.get(residente_id)

    def buscar_por_nombre(self, nombre: str) -> Optional[Residente]:
        """Busca un residente por nombre (coincidencia exacta, case-insensitive)."""
        nombre_lower = nombre.strip().lower()
        for residente in self._residentes.values():
            if residente.nombre.strip().lower() == nombre_lower:
                return residente
        return None

    def listar(self, solo_autorizados: bool = False) -> List[Residente]:
        """
        Lista todos los residentes registrados.
        """
        residentes = list(self._residentes.values())
        if solo_autorizados:
            residentes = [r for r in residentes if r.autorizado]
        return sorted(residentes, key=lambda r: r.nombre)

    def actualizar_autorizacion(
        self, residente_id: str, autorizado: bool
    ) -> Tuple[bool, str]:
        """
        Actualiza el estado de autorización de un residente.
        """
        residente = self.obtener(residente_id)
        if not residente:
            return False, f"No se encontró un residente con ID '{residente_id[:8]}...'."

        estado_anterior = residente.autorizado
        residente.autorizado = autorizado
        self._guardar_db()

        cambio = "AUTORIZADO" if autorizado else "NO AUTORIZADO"
        return True, (
            f"Residente '{residente.nombre}' actualizado: {cambio} "
            f"(anterior: {'AUTORIZADO' if estado_anterior else 'NO AUTORIZADO'})."
        )

    def eliminar_residente(self, residente_id: str) -> Tuple[bool, str]:
        """
        Elimina un residente y sus imágenes asociadas.
        """
        residente = self.obtener(residente_id)
        if not residente:
            return False, f"No se encontró un residente con ID '{residente_id[:8]}...'."

        nombre = residente.nombre

        for img in residente.imagenes:
            ruta_img = os.path.join(self.ruta_imagenes, img)
            if os.path.isfile(ruta_img):
                try:
                    os.remove(ruta_img)
                except OSError as e:
                    print(f"[ADVERTENCIA] No se pudo eliminar la imagen '{img}': {e}")

        del self._residentes[residente_id]
        self._guardar_db()

        return True, f"Residente '{nombre}' eliminado del sistema."

    def cargar_galeria(self) -> Tuple[List[np.ndarray], List[Residente]]:
        """
        Carga las codificaciones faciales de todos los residentes registrados.

        Cada codificación queda asociada al objeto Residente correspondiente,
        independientemente de si está autorizado o no.
        """
        from face_engine import cargar_imagen_rgb, codificar_rostro

        encodings = []
        residentes_validos = []

        for residente in self.listar():
            for imagen_nombre in residente.imagenes:
                ruta = os.path.join(self.ruta_imagenes, imagen_nombre)

                if not os.path.isfile(ruta):
                    print(
                        f"[ADVERTENCIA] No se encontró '{imagen_nombre}' "
                        f"para el residente '{residente.nombre}'."
                    )
                    continue

                try:
                    imagen_rgb = cargar_imagen_rgb(ruta)
                    encoding = codificar_rostro(imagen_rgb)

                    if encoding is None:
                        print(
                            f"[ADVERTENCIA] No se detectó un rostro válido "
                            f"en '{imagen_nombre}'."
                        )
                        continue

                    encodings.append(encoding)
                    residentes_validos.append(residente)

                except Exception as error:
                    print(
                        f"[ADVERTENCIA] No se pudo procesar "
                        f"'{imagen_nombre}': {error}"
                    )

        return encodings, residentes_validos


    def _generar_nombre_archivo_seguro(self, nombre_base: str, extension: str) -> str:
        """
        Genera un nombre de archivo seguro eliminando caracteres problemáticos.
        """
        caracteres_prohibidos = '<>:"/\\|?*'
        nombre_limpio = nombre_base
        for c in caracteres_prohibidos:
            nombre_limpio = nombre_limpio.replace(c, "_")

        nombre_limpio = nombre_limpio.lstrip(".")

        if not nombre_limpio:
            nombre_limpio = "residente"

        nombre_completo = f"{nombre_limpio}{extension}"

        ruta_completa = os.path.join(self.ruta_imagenes, nombre_completo)
        contador = 1
        while os.path.exists(ruta_completa):
            nombre_completo = f"{nombre_limpio}_{contador}{extension}"
            ruta_completa = os.path.join(self.ruta_imagenes, nombre_completo)
            contador += 1

        return nombre_completo

    def sincronizar_con_directorio(self) -> str:
        """
        Verifica que las imágenes referenciadas en la DB existan en el directorio
        y que las imágenes del directorio estén registradas.
        """
        mensajes = []

        for residente in list(self._residentes.values()):
            imagenes_faltantes = []
            for img in residente.imagenes:
                ruta = os.path.join(self.ruta_imagenes, img)
                if not os.path.isfile(ruta):
                    imagenes_faltantes.append(img)

            if imagenes_faltantes:
                for img in imagenes_faltantes:
                    residente.imagenes.remove(img)
                mensajes.append(
                    f"Residente '{residente.nombre}': eliminadas {len(imagenes_faltantes)} "
                    f"referencia(s) a imágenes inexistentes."
                )

        if os.path.isdir(self.ruta_imagenes):
            imagenes_registradas = set()
            for r in self._residentes.values():
                imagenes_registradas.update(r.imagenes)

            for archivo in sorted(os.listdir(self.ruta_imagenes)):
                if archivo.lower().endswith(EXTENSIONES_IMAGEN_VALIDAS):
                    if archivo not in imagenes_registradas:
                        mensajes.append(
                            f"Imagen '{archivo}' no asociada a ningún residente."
                        )

        if mensajes:
            self._guardar_db()
            return "\n".join(mensajes)
        return "La base de datos está sincronizada con el directorio de imágenes."