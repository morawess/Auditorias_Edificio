# Sistema de Acceso sin Llaves y Auditoría para Residentes (Reconocimiento Facial)

## Descripción del Proyecto
Este proyecto implementa una solución automatizada de **acceso sin llaves y auditoría de ingresos/egresos** para residentes de un edificio o condominio mediante reconocimiento facial. Consta de dos scripts principales que cumplen con todos los requerimientos técnicos solicitados:

1. **[`reconocimiento_facial.py`](./reconocimiento_facial.py)** *(Comparación y verificación entre dos imágenes)*:
   - Carga y procesa dos imágenes (una foto de control del residente registrado y una foto de prueba).
   - Convierte las imágenes de BGR a RGB para su correcta interpretación.
   - Localiza los rostros (`face_locations`), codifica sus características faciales (`face_encodings`) y los compara (`compare_faces`) calculando la distancia biométrica (`face_distance`).
   - Dibuja rectángulos sobre los rostros detectados y muestra en pantalla si hay coincidencia (`True`/`False`), la distancia facial redondeada y el estado de autorización (`ACCESO PERMITIDO` / `ACCESO DENEGADO`).

2. **[`control_acceso_residentes.py`](./control_acceso_residentes.py)** *(Control de acceso y auditoría en tiempo real por cámara web)*:
   - Carga dinámicamente la base de datos de fotos de residentes desde la carpeta [`Residentes/`](./Residentes/).
   - Extrae los nombres de los residentes y codifica sus características faciales.
   - Permite operar la cámara web tanto en **monitoreo continuo en tiempo real (streaming de video)** como en **captura única instantánea**.
   - Utiliza `numpy.argmin` para hallar la menor distancia facial contra la base de residentes.
   - Si hay coincidencia (`distancia <= 0.6`), muestra el recuadro verde con el nombre del residente y registra automáticamente el evento en [`registro_accesos.csv`](./registro_accesos.csv) detallando **Residente, Fecha, Hora, Tipo de Movimiento (`ENTRADA` / `SALIDA` alternado automáticamente) y Estado (`ACCESO SIN LLAVE`)**.

---

## Estructura del Proyecto

```text
Sistema de Asistencias/
├── Residentes/                    # Base de datos de imágenes de residentes del edificio
│   ├── Carl Sagan.jpg
│   ├── Cosmo Kramer.jpg
│   ├── Elaine Benes.jpg
│   ├── George Constanza.jpg
│   ├── Jerry Seinfeld.jpg
│   ├── Michio Kaku.jpg
│   └── Neil deGrasse Tyson.jpg
├── FotoA.jpg                      # Imagen de control para pruebas de verificación
├── FotoB.jpg                      # Imagen de prueba (misma persona que FotoA)
├── FotoC.jpg                      # Imagen de prueba (persona distinta a FotoA)
├── reconocimiento_facial.py       # Script 1: Comparación facial entre dos imágenes
├── control_acceso_residentes.py   # Script 2: Acceso sin llaves y auditoría por cámara web
├── registro_accesos.csv           # Registro histórico de auditoría (Entradas y Salidas)
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Documentación del proyecto
```

---

## Librerías Utilizadas
- **OpenCV (`cv2`)**: Procesamiento de imágenes, conversión de espacios de color, captura desde cámara web, dibujo de rectángulos y textos en pantalla.
- **Face Recognition (`face_recognition` / `fr`)**: Carga de imágenes, detección de ubicaciones de rostros, codificación biométrica y cálculo de distancias faciales.
- **OS (`os`)**: Manejo de rutas multiplataforma, listado de imágenes de residentes y extracción de nombres.
- **NumPy (`numpy`)**: Cálculo matemático para determinar el índice de menor distancia entre rostros (`numpy.argmin`).
- **Datetime (`datetime`)**: Obtención de la fecha y hora exacta para el registro de auditoría en CSV.

---

## Instalación de Dependencias

1. Verificar versión de Python y actualizar `pip`:
   ```bash
   python --version
   pip --version
   pip install --upgrade pip
   ```
2. Asegurarse de tener **CMake** instalado y agregado al `PATH` del sistema (requerido para compilar `dlib` / `face-recognition`).
3. Instalar las librerías requeridas:
   ```bash
   pip install -r requirements.txt
   ```
   *(O individualmente: `pip install opencv-python face-recognition numpy`)*

---

## Ejecución

### 1. Script de Comparación de Dos Imágenes
```bash
python reconocimiento_facial.py
```
Podrás elegir comparar `FotoA.jpg` con `FotoB.jpg` (coincidencia positiva), `FotoA.jpg` con `FotoC.jpg` (no coincidencia), o seleccionar cualquier imagen disponible.

### 2. Script de Control de Acceso y Auditoría en Tiempo Real
```bash
python control_acceso_residentes.py
```
Al iniciar, permite elegir entre:
- **Opción 1**: Monitoreo continuo en tiempo real (video en vivo; presionar `q` o `ESC` para salir).
- **Opción 2**: Captura única instantánea desde la cámara web.
