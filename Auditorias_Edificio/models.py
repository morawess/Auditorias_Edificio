"""
Modelos de datos del sistema de control de acceso facial.

Define las estructuras que representan residentes, resultados de identificación
y eventos de acceso. Utiliza dataclasses para claridad y compatibilidad con
serialización JSON.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional
import uuid


class ResultadoAcceso(Enum):
    """Resultado posible de un intento de acceso."""
    ACCESO_AUTORIZADO = "ACCESO_AUTORIZADO"
    ACCESO_DENEGADO_NO_AUTORIZADO = "ACCESO_DENEGADO_NO_AUTORIZADO"
    PERSONA_DESCONOCIDA = "PERSONA_DESCONOCIDA"
    ERROR_TECNICO = "ERROR_TECNICO"


class TipoMovimiento(Enum):
    """Tipo de movimiento registrado en el historial."""
    ENTRADA = "ENTRADA"
    SALIDA = "SALIDA"
    NO_APLICA = "N/A"


@dataclass
class Residente:
    """
    Representa a un residente registrado en el sistema.

    Attributes:
        id: Identificador único estable (UUID).
        nombre: Nombre completo del residente.
        autorizado: Si el residente tiene autorización vigente para ingresar.
        imagenes: Lista de nombres de archivos de imagen de referencia.
        fecha_registro: Fecha/hora de registro en formato ISO 8601.
        notas: Observaciones opcionales del administrador.
    """
    nombre: str
    id: str = ""
    autorizado: bool = True
    imagenes: List[str] = field(default_factory=list)
    fecha_registro: str = ""
    notas: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.fecha_registro:
            self.fecha_registro = datetime.now().astimezone().isoformat()

    def to_dict(self) -> dict:
        """Serializa el residente a un diccionario para almacenamiento JSON."""
        return {
            "id": self.id,
            "nombre": self.nombre,
            "autorizado": self.autorizado,
            "imagenes": list(self.imagenes),
            "fecha_registro": self.fecha_registro,
            "notas": self.notas,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Residente":
        """Crea un Residente a partir de un diccionario almacenado."""
        return cls(
            id=data.get("id", ""),
            nombre=data.get("nombre", ""),
            autorizado=data.get("autorizado", True),
            imagenes=data.get("imagenes", []),
            fecha_registro=data.get("fecha_registro", ""),
            notas=data.get("notas", ""),
        )


@dataclass
class ResultadoIdentificacion:
    """
    Resultado de un intento de identificación facial 1:N.

    Attributes:
        identificado: True si se encontró un candidato dentro del umbral.
        residente: Residente identificado (None si no se identificó).
        distancia: Distancia euclidiana al candidato más cercano.
        mensaje: Descripción del resultado para la interfaz.
    """
    identificado: bool = False
    residente: Optional[Residente] = None
    distancia: float = 1.0
    mensaje: str = ""

    @property
    def confianza_descripcion(self) -> str:
        """
        Descripción cualitativa del nivel de similitud.

        NOTA: Esto NO es una probabilidad de identidad. Es una interpretación
        orientativa de la distancia euclidiana del modelo. Un valor bajo de
        distancia indica mayor similitud entre los embeddings faciales, pero
        no garantiza que se trate de la misma persona.
        """
        if self.distancia < 0.35:
            return "Alta similitud"
        elif self.distancia < 0.45:
            return "Similitud media-alta"
        elif self.distancia < 0.55:
            return "Similitud media"
        elif self.distancia <= 0.6:
            return "Similitud baja (límite del umbral)"
        else:
            return "Similitud insuficiente"


@dataclass
class ResultadoVerificacion:
    """
    Resultado de una verificación facial 1:1 entre dos imágenes.

    Attributes:
        coincide: True si la distancia está dentro del umbral.
        distancia: Distancia euclidiana entre los dos embeddings.
        mensaje: Descripción del resultado.
    """
    coincide: bool = False
    distancia: float = 1.0
    mensaje: str = ""


@dataclass
class EventoAcceso:
    """
    Registro de un evento de acceso para el historial de auditoría.

    Attributes:
        fecha_hora: Marca temporal ISO 8601 con zona horaria.
        residente_id: UUID del residente (vacío si desconocido).
        nombre: Nombre del residente o "DESCONOCIDO".
        resultado: Tipo de resultado (ver ResultadoAcceso).
        movimiento: ENTRADA, SALIDA o N/A.
        distancia: Distancia facial calculada (vacío si no aplica).
        motivo: Descripción legible del motivo.
    """
    fecha_hora: str = ""
    residente_id: str = ""
    nombre: str = "DESCONOCIDO"
    resultado: str = ""
    movimiento: str = "N/A"
    distancia: str = ""
    motivo: str = ""

    def __post_init__(self):
        if not self.fecha_hora:
            self.fecha_hora = datetime.now().astimezone().isoformat()

    def to_csv_row(self) -> list:
        """Retorna los campos como lista para escritura CSV."""
        return [
            self.fecha_hora,
            self.residente_id,
            self.nombre,
            self.resultado,
            self.movimiento,
            self.distancia,
            self.motivo,
        ]

