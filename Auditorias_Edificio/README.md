# 🏢 Sistema de Control de Acceso Facial, Prueba de Vida y Auditoría

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Dlib / Face Recognition](https://img.shields.io/badge/Biometría-Dlib%20128D-orange.svg)](https://github.com/ageitgey/face_recognition)
[![Tkinter GUI](https://img.shields.io/badge/UI-Tkinter%20Dark-purple.svg)](https://docs.python.org/3/library/tkinter.html)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(6%2F6)-brightgreen.svg)](tests/)

Solución integral y modular de **control de acceso sin llaves, verificación biométrica facial y auditoría de seguridad** para edificios y condominios. El sistema combina visión por computadora en tiempo real, detección activa de vida (*anti-spoofing* contra fotos o videos), gestión de residentes con identificadores universales (UUIDv4) y una moderna interfaz gráfica de usuario.

---

## 🌟 Características Principales

* **🛡️ Prueba de Vida Activa (*Anti-Spoofing / Liveness*)**:
  - Evita la suplantación de identidad mediante fotos impresas o videos en pantallas.
  - Genera desafíos aleatorios e interactivos en vivo:
    - **Parpadeo Ocular**: Evalúa la relación de aspecto del ojo (**EAR** - *Eye Aspect Ratio*).
    - **Apertura Bucal**: Evalúa la relación de aspecto de los labios (**MAR** - *Mouth Aspect Ratio*).
  - Cuenta regresiva visual (10 segundos) y validación de una única persona en escena.
* **👁️ Reconocimiento Biométrico 1:N**:
  - Extracción de vectores faciales matemáticos de 128 dimensiones basados en redes neuronales convolucionales profundas (ResNet / dlib).
  - Cálculo de distancia euclidiana mínima en tiempo real contra la base de residentes registrados.
  - Alternancia automática inteligente entre **`ENTRADA`** y **`SALIDA`** según el último evento de cada persona.
  - Periodo de *cooldown* configurable (15 segundos) para evitar duplicación de eventos mientras el residente cruza la puerta.
* **🖥️ Interfaz Gráfica Unificada (GUI Dark Theme)**:
  - Desarrollada en Tkinter y Pillow con diseño oscuro moderno y fluido.
  - **Pestaña de Cámara en Vivo**: Streaming con recuadros interactivos, métricas HUD de liveness (EAR/MAR), slider dinámico de umbral, selector de dispositivo de cámara y botón para activar/desactivar la prueba de vida.
  - **Historial de Auditoría**: Tabla en tiempo real con código de colores (Verde: Autorizado, Azul: Salida, Rojo: Denegado) y acceso directo para abrir el archivo CSV en Excel.
  - **Pestaña de Comparación 1:1**: Cotejo estático entre dos fotografías con presets de prueba (`FotoA.jpg`, `FotoB.jpg`, `FotoC.jpg`), detección automática de rostros y diagnóstico visual.
* **👥 Gestión Persistente de Residentes (`residentes.json`)**:
  - Base de datos estructurada con UUIDs independientes del nombre de archivo.
  - Control de permisos administrativos: estado `autorizado: true/false` que permite inhabilitar un acceso temporalmente sin borrar la foto ni el historial.
  - Soporte de múltiples fotos por residente y sincronización automática con la carpeta física.
  - Manejo seguro de rutas y nombres con acentos o caracteres especiales en Windows.
* **📋 Auditoría Forense Inmutable (`registro_accesos.csv`)**:
  - Registro cronológico con marcas temporales bajo el estándar internacional **ISO 8601** con zona horaria local.
  - Registro de accesos permitidos, personas desconocidas, residentes no autorizados y fallos de prueba de vida.
  - Mecanismo de migración automática retrocompatible desde formatos anteriores con respaldo `.bak`.
* **⚡ Caché Biométrico de Arranque Instantáneo (`biometria_cache.pkl`)**:
  - Serialización de codificaciones faciales en disco.
  - Validación de marcas de tiempo (`mtime`): al abrir el sistema no se recalculan todas las fotos, logrando un inicio en milisegundos.

---

## 📂 Estructura del Proyecto

```text
Auditorias_Edificio/
├── .venv/                         # Entorno virtual de Python aislado
├── data/
│   ├── biometria_cache.pkl        # Caché serializado de embeddings faciales
│   └── residentes.json            # Base de datos documental de residentes (UUID, permisos)
├── Residentes/                    # Galería fotográfica de residentes del edificio
│   ├── Agustín Lecerasi.png
│   ├── Agustín Viñolo.png
│   ├── Carl Sagan.jpg
│   ├── Cosmo Kramer.jpg
│   ├── Elaine Benes.jpg
│   ├── George Constanza.jpg
│   ├── Jerry Seinfeld.jpg
│   ├── Juan Bautista Delprato.png
│   ├── Michio Kaku.jpg
│   ├── Mora Wessels.png
│   ├── Neil deGrasse Tyson.jpg
│   └── Pato.jpg
├── tests/
│   └── test_reconocimiento_facial.py # Suite de pruebas unitarias automatizadas con mocks
├── access_control.py              # Controlador de accesos, autorización y cooldown
├── camera.py                      # Gestor de contexto seguro de cámara web (CameraManager)
├── config.py                      # Constantes y parámetros centralizados del sistema
├── control_acceso_residentes.py   # Punto de entrada por terminal (CLI con menú interactivo)
├── event_logger.py                # Registrador inmutable y auditoría forense en CSV
├── face_engine.py                 # Motor matemático de visión artificial y biometría
├── interfaz_grafica.py            # Aplicación de Interfaz Gráfica de Usuario (GUI)
├── liveness.py                    # Motor de Prueba de Vida (algoritmos EAR y MAR)
├── models.py                      # Modelos de datos, dataclasses y enumeradores
├── prueba_vida.py                 # Script de calibración y prueba aislada de liveness
├── reconocimiento_facial.py       # Script terminal de verificación facial estática 1:1
├── registro_accesos.csv           # Archivo de auditoría de accesos del edificio
├── requirements.txt               # Lista de librerías y dependencias
├── run_control_acceso.bat         # Lanzador rápido para consola / terminal
├── run_interfaz.bat               # Lanzador rápido para la Interfaz Gráfica (GUI)
└── run_reconocimiento.bat         # Lanzador rápido para verificación 1:1
```

---

## 🚀 Instalación y Puesta en Marcha

### Requisitos Previos
- **Python 3.10 o superior** instalado en el sistema.
- **CMake** instalado y agregado al `PATH` (necesario para compilar las librerías C++ de `dlib`).

### Configuración del Entorno

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/morawess/Auditorias_Edificio.git
   cd Auditorias_Edificio/Auditorias_Edificio
   ```

2. **Crear y activar el entorno virtual**:
   ```bash
   # En Windows
   python -m venv .venv
   .venv\Scripts\activate

   # En Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 💻 Modos de Ejecución

### 1. Interfaz Gráfica Completa (Recomendado)
Haz doble clic en **`run_interfaz.bat`** o ejecuta desde la terminal:
```bash
python interfaz_grafica.py
```
- **Pestaña "Control de Acceso en Vivo"**: Activa la cámara web, evalúa residentes, ejecuta pruebas de vida y visualiza el historial de auditoría.
- **Pestaña "Comparación Facial 1:1"**: Selecciona fotos estáticas o presets para analizar compatibilidad facial.

### 2. Terminal CLI / Modo Tótem
Haz doble clic en **`run_control_acceso.bat`** o ejecuta:
```bash
python control_acceso_residentes.py
```
Desplegará un menú interactivo:
- **`[1] Monitoreo continuo en Tiempo Real (con Prueba de Vida)`**: Flujo con desafío interactivo de parpadeo/boca antes de abrir la puerta.
- **`[2] Captura única instantánea`**: Toma una fotografía fija y valida de inmediato sin liveness.

### 3. Calibrador de Prueba de Vida Aislado
Para ajustar la sensibilidad de la cámara o comprobar cómo responde el parpadeo y la boca:
```bash
python prueba_vida.py
```

### 4. Verificación Facial 1:1 por Consola
Haz doble clic en **`run_reconocimiento.bat`** o ejecuta:
```bash
python reconocimiento_facial.py
```

### 5. Ejecutar la Suite de Pruebas Unitarias
El proyecto incluye pruebas automatizadas con emulación (*mocking*) que verifican la lógica del motor sin requerir cámaras físicas:
```bash
python -m unittest discover -s tests -v
```

---

## ⚙️ Conceptos Clave del Sistema

### ¿Qué es el Umbral de Distancia y cómo ajustarlo?
El motor biométrico traduce cada rostro a un espacio vectorial de 128 dimensiones. La comparación calcula la **distancia euclidiana** entre dos vectores:
- **Distancia $\le$ Umbral**: El sistema confirma que se trata de la misma persona y concede el paso.
- **Distancia $>$ Umbral**: El sistema considera que no hay coincidencia suficiente y deniega el acceso.

| Rango de Umbral | Comportamiento | Escenario de Uso |
| :---: | :--- | :--- |
| **`0.45 - 0.50`** | **Alta Seguridad (Muy Estricto)** | Accesos bancarios, salas de servidores o áreas restringidas. Minimiza falsos positivos a casi cero. |
| **`0.55 - 0.60`** | **Equilibrio Estándar (Recomendado)** | Acceso principal a edificios residenciales. Buena tolerancia a cambios de ángulo e iluminación sin perder seguridad. |
| **`0.65 - 0.70`** | **Permisivo** | Zonas con cámaras de baja resolución o iluminación nocturna irregular. |

### ¿Cómo funciona la Prueba de Vida (*Anti-Spoofing*)?
El sistema extrae los 68 puntos faciales de referencia (*landmarks*) y evalúa fotograma a fotograma:
1. **EAR (*Eye Aspect Ratio*)**: Calcula la distancia vertical entre párpados dividida por el ancho del ojo. Detecta cuando el valor cae bruscamente (`< 0.20`) y vuelve a subir (`> 0.24`), confirmando un parpadeo biológico real.
2. **MAR (*Mouth Aspect Ratio*)**: Calcula la distancia vertical entre los labios superior e inferior dividida por el ancho de la boca. Detecta apertura voluntaria (`> 0.54`).
3. Si el usuario no cumple el desafío dentro de los **10 segundos**, el acceso se bloquea por tiempo agotado y se audita el incidente como posible intento de suplantación.

---

## 👥 Equipo de Desarrollo

Proyecto desarrollado y unificado colaborativamente:
- **Mora Wessels**
- **Agustín Viñolo**
- **Juan Bautista Delprato**
- **Agustín Lecerasi**
