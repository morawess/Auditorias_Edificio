"""
Control de acceso facial.

Implementa la lógica de decisión de acceso: recibe el resultado de la
identificación facial y determina si el acceso es autorizado o denegado,
registrando el evento correspondiente.

Principio de seguridad:
    Cuando el sistema no puede completar una verificación fiable, falla de
    forma segura y NO autoriza el ingreso automáticamente.
"""

from datetime import datetime
from typing import Dict, Optional, Tuple

from models import (
    EventoAcceso,
    Residente,
    ResultadoAcceso,
    ResultadoIdentificacion,
    TipoMovimiento,
)


class AccessController:
    """
    Controlador de acceso al edificio.

    Coordina la identificación facial, la verificación de autorización
    y el registro de eventos. Mantiene un cooldown por residente para
    evitar registros duplicados durante el monitoreo continuo.
    """

    def __init__(self, event_logger, tiempo_cooldown: float = 15.0):
        """
        Args:
            event_logger: Instancia de EventLogger para registrar eventos.
            tiempo_cooldown: Segundos mínimos entre registros consecutivos
                            del mismo residente.
        """
        self.event_logger = event_logger
        self.tiempo_cooldown = tiempo_cooldown
        self._ultimos_registros: Dict[str, float] = {}

    def procesar_identificacion(
        self, resultado: ResultadoIdentificacion
    ) -> Tuple[bool, str, Optional[EventoAcceso]]:
        """
        Procesa el resultado de una identificación facial y determina
        si se concede el acceso.

        Flujo de decisión:
        1. Si no se identificó a nadie → PERSONA_DESCONOCIDA → acceso denegado.
        2. Si se identificó pero no está autorizado → ACCESO_DENEGADO → acceso denegado.
        3. Si se identificó y está autorizado → ACCESO_AUTORIZADO → acceso concedido.

        Args:
            resultado: ResultadoIdentificacion del motor facial.

        Returns:
            Tupla (acceso_concedido, mensaje, evento_registrado).
        """
        if not resultado.identificado or resultado.residente is None:
            # Persona desconocida
            evento = self.event_logger.registrar_acceso_denegado(
                motivo=resultado.mensaje,
                distancia=resultado.distancia,
            )
            mensaje = (
                "⛔ ACCESO DENEGADO — Persona no identificada.\n"
                f"   {resultado.mensaje}\n"
                "   Contacte al encargado del edificio o utilice un método "
                "alternativo de verificación."
            )
            return False, mensaje, evento

        residente = resultado.residente

        if not residente.autorizado:
            # Residente identificado pero no autorizado
            evento = self.event_logger.registrar_acceso_denegado(
                residente_id=residente.id,
                nombre=residente.nombre,
                distancia=resultado.distancia,
                motivo=(
                    f"Residente '{residente.nombre}' identificado pero sin "
                    f"autorización vigente para ingresar."
                ),
            )
            mensaje = (
                f"⛔ ACCESO DENEGADO — Residente no autorizado.\n"
                f"   Identidad reconocida: {residente.nombre}\n"
                f"   Estado de autorización: NO AUTORIZADO\n"
                f"   Contacte al administrador del edificio para gestionar "
                f"su autorización."
            )
            return False, mensaje, evento

        # Residente identificado y autorizado — verificar cooldown
        ahora = datetime.now().timestamp()
        ultimo = self._ultimos_registros.get(residente.id, 0.0)

        if (ahora - ultimo) < self.tiempo_cooldown:
            # Dentro del cooldown, no registrar de nuevo
            mensaje = (
                f"✅ ACCESO AUTORIZADO — {residente.nombre}\n"
                f"   (Registro reciente, sin duplicar evento)"
            )
            return True, mensaje, None

        # Determinar movimiento (ENTRADA/SALIDA)
        movimiento = self.event_logger.determinar_movimiento(residente.id)

        # Registrar acceso autorizado
        evento = self.event_logger.registrar_acceso_autorizado(
            residente_id=residente.id,
            nombre=residente.nombre,
            movimiento=movimiento,
            distancia=resultado.distancia,
        )

        self._ultimos_registros[residente.id] = ahora

        mensaje = (
            f"✅ ACCESO AUTORIZADO — {residente.nombre}\n"
            f"   Movimiento: {movimiento.value}\n"
            f"   Similitud: {resultado.confianza_descripcion} "
            f"(distancia: {resultado.distancia:.4f})"
        )
        return True, mensaje, evento

    def procesar_error_tecnico(self, descripcion: str) -> EventoAcceso:
        """
        Registra un error técnico y retorna el evento.

        Cuando ocurre un error técnico, el acceso NO se concede.

        Args:
            descripcion: Descripción del error ocurrido.

        Returns:
            EventoAcceso registrado.
        """
        evento = self.event_logger.registrar_error_tecnico(
            motivo=descripcion,
        )
        return evento

    def reiniciar_cooldowns(self):
        """Limpia el registro de cooldowns."""
        self._ultimos_registros.clear()

