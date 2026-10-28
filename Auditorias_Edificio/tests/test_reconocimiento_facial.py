"""Pruebas unitarias del flujo de comparación del Sprint 2.

Estas pruebas simulan el motor facial para validar el flujo y los errores sin
necesitar cámara ni fotografías reales. No miden la precisión del algoritmo.
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import reconocimiento_facial as app
from models import ResultadoVerificacion


CAJA_ROSTRO = (10, 80, 90, 20)


class ComparacionImagenesTests(unittest.TestCase):
    def setUp(self):
        self.validar = patch.object(
            app.motor,
            "validar_archivo_imagen",
            return_value=(True, "Imagen válida."),
        )
        self.cargar = patch.object(
            app.motor,
            "cargar_imagen_rgb",
            return_value=np.zeros((100, 100, 3), dtype=np.uint8),
        )
        self.detectar = patch.object(
            app.motor,
            "detectar_rostros",
            side_effect=[[CAJA_ROSTRO], [CAJA_ROSTRO]],
        )
        self.codificar = patch.object(
            app.motor,
            "codificar_rostro",
            return_value=np.zeros(128, dtype=np.float64),
        )
        self.verificar = patch.object(
            app.motor,
            "verificar_1a1",
            return_value=ResultadoVerificacion(
                coincide=True,
                distancia=0.31,
                mensaje="Coincidencia detectada.",
            ),
        )

        self.mock_validar = self.validar.start()
        self.addCleanup(self.validar.stop)
        self.mock_cargar = self.cargar.start()
        self.addCleanup(self.cargar.stop)
        self.mock_detectar = self.detectar.start()
        self.addCleanup(self.detectar.stop)
        self.mock_codificar = self.codificar.start()
        self.addCleanup(self.codificar.stop)
        self.mock_verificar = self.verificar.start()
        self.addCleanup(self.verificar.stop)

    def test_compara_dos_imagenes_y_devuelve_resultado(self):
        resultado = app.comparar_imagenes(
            "referencia.jpg",
            "prueba.jpg",
            umbral=0.5,
            mostrar=False,
        )

        self.assertTrue(resultado.coincide)
        self.assertAlmostEqual(resultado.distancia, 0.31)
        self.assertEqual(self.mock_detectar.call_count, 2)
        self.assertEqual(self.mock_codificar.call_count, 2)
        self.mock_verificar.assert_called_once_with(
            self.mock_codificar.return_value,
            self.mock_codificar.return_value,
            umbral=0.5,
        )

    def test_rechaza_imagen_sin_rostros(self):
        self.mock_detectar.side_effect = [[]]

        with self.assertRaisesRegex(ValueError, "no se detectó ningún rostro"):
            app.comparar_imagenes("referencia.jpg", "prueba.jpg", mostrar=False)

        self.mock_verificar.assert_not_called()

    def test_rechaza_imagen_con_mas_de_un_rostro(self):
        self.mock_detectar.side_effect = [[CAJA_ROSTRO, CAJA_ROSTRO]]

        with self.assertRaisesRegex(ValueError, "exactamente uno"):
            app.comparar_imagenes("referencia.jpg", "prueba.jpg", mostrar=False)

        self.mock_verificar.assert_not_called()

    def test_rechaza_archivo_invalido(self):
        self.mock_validar.return_value = (False, "El archivo no existe.")

        with self.assertRaisesRegex(ValueError, "El archivo no existe"):
            app.comparar_imagenes("inexistente.jpg", "prueba.jpg", mostrar=False)

        self.mock_cargar.assert_not_called()

    def test_rechaza_umbral_negativo(self):
        with self.assertRaisesRegex(ValueError, "no puede ser negativo"):
            app.comparar_imagenes("referencia.jpg", "prueba.jpg", umbral=-0.1)

    def test_rechaza_umbral_no_finito(self):
        with self.assertRaisesRegex(ValueError, "número finito"):
            app.comparar_imagenes("referencia.jpg", "prueba.jpg", umbral=float("nan"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
