"""
Registro de eventos de acceso (auditoría).

Gestiona la persistencia y consulta de los intentos de acceso en formato CSV.
Utiliza el módulo csv de la biblioteca estándar para garantizar un formato
consistente y resistente a caracteres especiales en los campos.

Los registros nunca se sobrescriben: cada evento se agrega al final del archivo.
Los registros incluyen una marca temporal ISO 8601 con zona horaria.
"""

import csv
import os
import shutil
from datetime import datetime
from typing import List, Optional

from config import ARCHIVO_REGISTRO_EVENTOS, CAMPOS_CSV
from models import EventoAcceso, ResultadoAcceso, TipoMovimiento


class EventLogger:
    """
    Registrador de eventos de acceso al edificio.

    Mantiene un archivo CSV con el historial de intentos de acceso,
    incluyendo accesos autorizados, denegados, personas desconocidas
    y errores técnicos.
    """

    def __init__(self, ruta_csv: str = ARCHIVO_REGISTRO_EVENTOS):
        self.ruta_csv = ruta_csv
        self._inicializar()

    def _inicializar(self):
        """
        Verifica el archivo CSV y migra del formato anterior si es necesario.
        """
        if not os.path.exists(self.ruta_csv):
            self._crear_archivo_nuevo()
            return

        # Verificar si es el formato anterior y migrar
        try:
            with open(self.ruta_csv, "r", encoding="utf-8") as f:
                primera_linea = f.readline().strip()

            if primera_linea and "fecha_hora" not in primera_linea:
                self._migrar_formato_anterior()
        except OSError:
            self._crear_archivo_nuevo()

    def _crear_archivo_nuevo(self):
        """Crea un archivo CSV nuevo con el encabezado correcto."""
        try:
            with open(self.ruta_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(CAMPOS_CSV)
        except OSError as e:
            print(f"[ERROR] No se pudo crear el archivo de registro: {e}")

    def _migrar_formato_anterior(self):
        """
        Migra el formato CSV anterior al nuevo formato, preservando los datos
        existentes.

        Formato anterior: residente, fecha, hora, movimiento, estado
        Formato nuevo: fecha_hora, residente_id, nombre, resultado, movimiento, distancia, motivo
        """
        print("[INFO] Migrando registro de accesos al nuevo formato...")

        try:
            # Leer registros antiguos
            registros_antiguos = []
            with open(self.ruta_csv, "r", encoding="utf-8") as f:
                lineas = f.readlines()

            for linea in lineas[1:]:  # Omitir encabezado antiguo
                partes = [campo.strip() for campo in linea.strip().split(",")]
                if len(partes) >= 5:
                    nombre = partes[0]
                    fecha = partes[1]
                    hora = partes[2]
                    movimiento = partes[3]
                    estado = partes[4]

                    # Construir timestamp ISO 8601 a partir de fecha + hora
                    try:
                        dt = datetime.strptime(f"{fecha} {hora}", "%Y-%m-%d %H:%M:%S")
                        fecha_hora = dt.astimezone().isoformat()
                    except ValueError:
                        fecha_hora = f"{fecha}T{hora}"

                    evento = EventoAcceso(
                        fecha_hora=fecha_hora,
                        residente_id="",  # No disponible en formato anterior
                        nombre=nombre,
                        resultado=ResultadoAcceso.ACCESO_AUTORIZADO.value,
                        movimiento=movimiento,
                        distancia="",
                        motivo=f"Migrado del registro anterior ({estado})",
                    )
                    registros_antiguos.append(evento)

            # Respaldar archivo anterior
            backup = self.ruta_csv + ".bak"
            shutil.copy2(self.ruta_csv, backup)
            print(f"[INFO] Respaldo del formato anterior guardado en: {os.path.basename(backup)}")

            # Escribir nuevo formato
            self._crear_archivo_nuevo()
            for evento in registros_antiguos:
                self._escribir_evento(evento)

            print(f"[INFO] {len(registros_antiguos)} registro(s) migrados al nuevo formato.")

        except Exception as e:
            print(f"[ERROR] Error durante la migración: {e}")
            print("[INFO] Se creará un archivo de registro nuevo.")
            self._crear_archivo_nuevo()

    def registrar_acceso_autorizado(
        self,
        residente_id: str,
        nombre: str,
        movimiento: TipoMovimiento,
        distancia: float,
    ) -> EventoAcceso:
        """
        Registra un acceso autorizado.

        Args:
            residente_id: UUID del residente.
            nombre: Nombre del residente.
            movimiento: ENTRADA o SALIDA.
            distancia: Distancia facial calculada.

        Returns:
            EventoAcceso registrado.
        """
        evento = EventoAcceso(
            residente_id=residente_id,
            nombre=nombre,
            resultado=ResultadoAcceso.ACCESO_AUTORIZADO.value,
            movimiento=movimiento.value,
            distancia=f"{distancia:.4f}",
            motivo="Residente autorizado identificado por reconocimiento facial.",
        )
        self._escribir_evento(evento)
        return evento

    def registrar_acceso_denegado(
        self,
        residente_id: str = "",
        nombre: str = "DESCONOCIDO",
        distancia: float = 0.0,
        motivo: str = "",
    ) -> EventoAcceso:
        """
        Registra un intento de acceso denegado.

        Args:
            residente_id: UUID del residente (vacío si desconocido).
            nombre: Nombre del residente o "DESCONOCIDO".
            distancia: Distancia facial (0 si no aplica).
            motivo: Motivo del rechazo.

        Returns:
            EventoAcceso registrado.
        """
        resultado = (
            ResultadoAcceso.ACCESO_DENEGADO_NO_AUTORIZADO.value
            if residente_id
            else ResultadoAcceso.PERSONA_DESCONOCIDA.value
        )
        evento = EventoAcceso(
            residente_id=residente_id,
            nombre=nombre,
            resultado=resultado,
            movimiento=TipoMovimiento.NO_APLICA.value,
            distancia=f"{distancia:.4f}" if distancia > 0 else "",
            motivo=motivo or "Acceso denegado.",
        )
        self._escribir_evento(evento)
        return evento

    def registrar_error_tecnico(self, motivo: str) -> EventoAcceso:
        """
        Registra un error técnico que impidió completar la verificación.

        Args:
            motivo: Descripción del error.

        Returns:
            EventoAcceso registrado.
        """
        evento = EventoAcceso(
            resultado=ResultadoAcceso.ERROR_TECNICO.value,
            movimiento=TipoMovimiento.NO_APLICA.value,
            motivo=motivo,
        )
        self._escribir_evento(evento)
        return evento

    def determinar_movimiento(self, residente_id: str) -> TipoMovimiento:
        """
        Determina si el próximo movimiento del residente es ENTRADA o SALIDA
        basándose en su último registro.

        Args:
            residente_id: UUID del residente.

        Returns:
            TipoMovimiento correspondiente.
        """
        historial = self.obtener_historial(residente_id=residente_id)

        # Buscar último movimiento válido (ENTRADA o SALIDA)
        for evento in reversed(historial):
            if evento.movimiento == TipoMovimiento.ENTRADA.value:
                return TipoMovimiento.SALIDA
            elif evento.movimiento == TipoMovimiento.SALIDA.value:
                return TipoMovimiento.ENTRADA

        return TipoMovimiento.ENTRADA

    def obtener_historial(
        self,
        residente_id: Optional[str] = None,
        resultado: Optional[str] = None,
        limite: int = 0,
    ) -> List[EventoAcceso]:
        """
        Consulta el historial de eventos de acceso.

        Args:
            residente_id: Filtrar por UUID de residente (None = todos).
            resultado: Filtrar por tipo de resultado (None = todos).
            limite: Número máximo de registros (0 = sin límite).

        Returns:
            Lista de EventoAcceso ordenada cronológicamente.
        """
        eventos = []

        if not os.path.isfile(self.ruta_csv):
            return eventos

        try:
            with open(self.ruta_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Aplicar filtros
                    if residente_id and row.get("residente_id", "") != residente_id:
                        continue
                    if resultado and row.get("resultado", "") != resultado:
                        continue

                    evento = EventoAcceso(
                        fecha_hora=row.get("fecha_hora", ""),
                        residente_id=row.get("residente_id", ""),
                        nombre=row.get("nombre", ""),
                        resultado=row.get("resultado", ""),
                        movimiento=row.get("movimiento", ""),
                        distancia=row.get("distancia", ""),
                        motivo=row.get("motivo", ""),
                    )
                    eventos.append(evento)

        except (OSError, csv.Error) as e:
            print(f"[ERROR] No se pudo leer el historial de accesos: {e}")

        if limite > 0:
            eventos = eventos[-limite:]

        return eventos

    def _escribir_evento(self, evento: EventoAcceso):
        """Escribe un evento al archivo CSV."""
        try:
            with open(self.ruta_csv, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(evento.to_csv_row())
        except OSError as e:
            print(f"[ERROR] No se pudo registrar el evento de acceso: {e}")

