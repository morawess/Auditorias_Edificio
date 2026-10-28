"""
Interfaz Gráfica Unificada: Control de Acceso y Verificación Facial 1:1
------------------------------------------------------------------------
Proporciona una interfaz visual, moderna y fácil de usar para:
1. Control de Acceso y Auditoría en tiempo real mediante Cámara Web (reconocimiento facial de residentes y registro CSV).
2. Verificación y Comparación Facial 1:1 entre dos imágenes estáticas con umbral configurable y métrica de distancia.
"""

import os
import pickle
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, List, Tuple
from PIL import Image, ImageTk
import numpy as np
import cv2
import face_recognition as fr

# Importar configuraciones y módulos existentes del proyecto
from config import (
    DIRECTORIO_BASE,
    DIRECTORIO_RESIDENTES,
    ARCHIVO_REGISTRO_EVENTOS,
    UMBRAL_DISTANCIA,
    EXTENSIONES_IMAGEN_VALIDAS,
    TIEMPO_ENTRE_REGISTROS_SEG,
)
import control_acceso_residentes as car
import face_engine as motor

ARCHIVO_REGISTRO_CSV = ARCHIVO_REGISTRO_EVENTOS

# Bloqueo de concurrencia para operaciones de dlib / face_recognition
dlib_lock = threading.Lock()



def cargar_imagen_segura_rgb(ruta: str) -> Optional[np.ndarray]:
    """Carga una imagen en formato RGB soportando rutas con caracteres especiales en Windows."""
    try:
        datos = np.fromfile(ruta, dtype=np.uint8)
        if datos.size == 0:
            return None
        bgr = cv2.imdecode(datos, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


class AppControlAccesoGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Control de Acceso y Verificación Facial — Edificio")

        # Ajuste dinámico de dimensiones adaptado a pantallas de laptops (ej. 1366x768)
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        ancho = min(1140, max(850, sw - 50))
        alto = min(660, max(480, sh - 90))
        x = max(0, (sw - ancho) // 2)
        y = max(10, (sh - alto - 50) // 2)
        self.root.geometry(f"{ancho}x{alto}+{x}+{y}")
        self.root.minsize(900, 480)
        self.root.configure(bg="#0f172a")

        # Estado del módulo de cámara y acceso
        self.captura: Optional[cv2.VideoCapture] = None
        self.camara_activa = False
        self.frame_count = 0
        self.ultimas_caras: List[Tuple] = []
        self.ultimas_etiquetas: List[Tuple[str, bool, float]] = []
        self.ultimos_registros: dict = {}

        # Base de residentes
        self.imagenes_residentes = []
        self.nombres_residentes = []
        self.lista_codificada = []
        self.nombres_validos = []

        # Estado del módulo 1:1
        self.ruta_img_a: Optional[str] = None
        self.ruta_img_b: Optional[str] = None
        self.img_rgb_a: Optional[np.ndarray] = None
        self.img_rgb_b: Optional[np.ndarray] = None
        self.encoding_a: Optional[np.ndarray] = None
        self.encoding_b: Optional[np.ndarray] = None
        self.caja_a: Optional[Tuple[int, int, int, int]] = None
        self.caja_b: Optional[Tuple[int, int, int, int]] = None

        # Configurar estilos visuales (Tema Dark Moderno)
        self._configurar_estilos()

        # Construir Interfaz
        self._construir_ui()

        # Protocolo de cierre seguro
        self.root.protocol("WM_DELETE_WINDOW", self._al_cerrar)

        # Cargar base de residentes en un hilo para no bloquear la interfaz inicial
        self.root.after(100, self._iniciar_carga_residentes)

    def _configurar_estilos(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Notebook (Pestañas)
        style.configure(
            "TNotebook",
            background="#0f172a",
            borderwidth=0,
            tabmargins=[10, 8, 10, 0],
        )
        style.configure(
            "TNotebook.Tab",
            background="#1e293b",
            foreground="#94a3b8",
            padding=[18, 9],
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#2563eb")],
            foreground=[("selected", "#ffffff")],
        )

        # Treeview (Tabla de auditoría)
        style.configure(
            "Treeview",
            background="#1e293b",
            foreground="#f8fafc",
            fieldbackground="#1e293b",
            rowheight=26,
            font=("Segoe UI", 9),
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            background="#0f172a",
            foreground="#94a3b8",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
        )
        style.map(
            "Treeview.Heading",
            background=[("active", "#1e293b")],
            foreground=[("active", "#ffffff")],
        )
        style.map(
            "Treeview",
            background=[("selected", "#3b82f6")],
            foreground=[("selected", "#ffffff")],
        )

    def _construir_ui(self):
        # Cabecera superior
        header = tk.Frame(self.root, bg="#1e293b", height=60)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        titulo_box = tk.Frame(header, bg="#1e293b")
        titulo_box.pack(side=tk.LEFT, padx=20, pady=10)

        lbl_logo = tk.Label(
            titulo_box,
            text="[ACCESO]",
            font=("Segoe UI", 12, "bold"),
            bg="#1e293b",
            fg="#60a5fa",
        )
        lbl_logo.pack(side=tk.LEFT, padx=(0, 10))

        lbl_tit = tk.Label(
            titulo_box,
            text="SISTEMA DE ACCESO Y AUDITORÍA RESIDENCIAL",
            font=("Segoe UI", 12, "bold"),
            bg="#1e293b",
            fg="#f8fafc",
        )
        lbl_tit.pack(anchor="w")

        lbl_sub = tk.Label(
            titulo_box,
            text="Control biométrico facial en tiempo real y verificación de identidad 1:1",
            font=("Segoe UI", 9),
            bg="#1e293b",
            fg="#94a3b8",
        )
        lbl_sub.pack(anchor="w")

        self.lbl_estado_bd = tk.Label(
            header,
            text="Cargando residentes...",
            font=("Segoe UI", 9, "bold"),
            bg="#334155",
            fg="#fbbf24",
            padx=12,
            pady=4,
        )
        self.lbl_estado_bd.pack(side=tk.RIGHT, padx=20)

        # Contenedor de Pestañas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        # Pestaña 1: Cámara y Control de Acceso
        self.tab_camara = tk.Frame(self.notebook, bg="#0f172a")
        self.notebook.add(self.tab_camara, text="  Control de Acceso en Vivo (Cámara)  ")
        self._construir_tab_camara()

        # Pestaña 2: Comparación Facial 1:1
        self.tab_comparacion = tk.Frame(self.notebook, bg="#0f172a")
        self.notebook.add(self.tab_comparacion, text="  Comparación Facial 1:1 (Fotos)  ")
        self._construir_tab_comparacion()

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1: CÁMARA Y CONTROL DE ACCESO
    # ──────────────────────────────────────────────────────────────────────────
    def _construir_tab_camara(self):
        panel_principal = tk.Frame(self.tab_camara, bg="#0f172a")
        panel_principal.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Columna Izquierda: Video en vivo y controles de cámara
        col_video = tk.Frame(panel_principal, bg="#0f172a")
        col_video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        # Barra de Controles de Cámara (siempre visible ARRIBA del video)
        controles_cam = tk.Frame(col_video, bg="#1e293b", pady=8, padx=10, bd=1, relief="solid")
        controles_cam.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))

        self.btn_toggle_cam = tk.Button(
            controles_cam,
            text="▶ Iniciar Cámara",
            font=("Segoe UI", 10, "bold"),
            bg="#10b981",
            fg="#ffffff",
            activebackground="#059669",
            activeforeground="#ffffff",
            relief="flat",
            padx=16,
            pady=6,
            cursor="hand2",
            command=self._toggle_camara,
        )
        self.btn_toggle_cam.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_captura_unica = tk.Button(
            controles_cam,
            text="Captura Única",
            font=("Segoe UI", 9, "bold"),
            bg="#3b82f6",
            fg="#ffffff",
            activebackground="#2563eb",
            activeforeground="#ffffff",
            relief="flat",
            padx=12,
            pady=6,
            cursor="hand2",
            command=self._ejecutar_captura_unica,
        )
        self.btn_captura_unica.pack(side=tk.LEFT, padx=(0, 15))

        # Selector de índice de cámara
        tk.Label(
            controles_cam,
            text="Cámara:",
            bg="#1e293b",
            fg="#94a3b8",
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))

        self.var_cam_index = tk.IntVar(value=0)
        spin_cam = ttk.Spinbox(
            controles_cam,
            from_=0,
            to=5,
            width=3,
            textvariable=self.var_cam_index,
        )
        spin_cam.pack(side=tk.LEFT, padx=(0, 15))

        # Slider de Umbral
        tk.Label(
            controles_cam,
            text="Umbral:",
            bg="#1e293b",
            fg="#94a3b8",
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))

        self.var_umbral_cam = tk.DoubleVar(value=0.60)
        slider_umbral = ttk.Scale(
            controles_cam,
            from_=0.40,
            to=0.75,
            variable=self.var_umbral_cam,
            orient=tk.HORIZONTAL,
            length=100,
            command=lambda v: self.lbl_val_umbral_cam.config(text=f"{float(v):.2f}"),
        )
        slider_umbral.pack(side=tk.LEFT, padx=(0, 6))

        self.lbl_val_umbral_cam = tk.Label(
            controles_cam,
            text="0.60",
            bg="#1e293b",
            fg="#60a5fa",
            font=("Segoe UI", 9, "bold"),
            width=4,
        )
        self.lbl_val_umbral_cam.pack(side=tk.LEFT)

        # Marco del visor de video (debajo de los controles)
        visor_card = tk.Frame(col_video, bg="#1e293b", bd=1, relief="solid")
        visor_card.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas_video = tk.Canvas(
            visor_card,
            bg="#020617",
            highlightthickness=0,
        )
        self.canvas_video.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.canvas_video.bind("<Configure>", lambda e: self._al_redimensionar_canvas())
        self.root.after(100, lambda: self._dibujar_placeholder_camara("Cámara inactiva\nPresione 'Iniciar Cámara' para comenzar"))

        # Columna Derecha: Tarjeta de Estado y Tabla de Auditoría
        col_derecha = tk.Frame(panel_principal, bg="#0f172a", width=380)
        col_derecha.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(8, 0))
        col_derecha.pack_propagate(False)

        # Tarjeta 1: Estado del Acceso en Tiempo Real
        card_estado = tk.Frame(col_derecha, bg="#1e293b", bd=1, relief="solid", padx=12, pady=10)
        card_estado.pack(fill=tk.X, pady=(0, 8))

        lbl_tit_card = tk.Label(
            card_estado,
            text="ESTADO DEL ACCESO EN TIEMPO REAL",
            font=("Segoe UI", 9, "bold"),
            bg="#1e293b",
            fg="#94a3b8",
        )
        lbl_tit_card.pack(anchor="w", pady=(0, 6))

        self.badge_estado = tk.Label(
            card_estado,
            text="EN ESPERA DE ROSTRO",
            font=("Segoe UI", 12, "bold"),
            bg="#334155",
            fg="#f8fafc",
            pady=8,
        )
        self.badge_estado.pack(fill=tk.X, pady=(0, 8))

        # Detalles del residente
        grid_detalles = tk.Frame(card_estado, bg="#1e293b")
        grid_detalles.pack(fill=tk.X)

        tk.Label(grid_detalles, text="Residente:", font=("Segoe UI", 9), bg="#1e293b", fg="#94a3b8").grid(row=0, column=0, sticky="w", pady=2)
        self.lbl_nombre_det = tk.Label(grid_detalles, text="—", font=("Segoe UI", 10, "bold"), bg="#1e293b", fg="#ffffff")
        self.lbl_nombre_det.grid(row=0, column=1, sticky="w", padx=6, pady=2)

        tk.Label(grid_detalles, text="Movimiento:", font=("Segoe UI", 9), bg="#1e293b", fg="#94a3b8").grid(row=1, column=0, sticky="w", pady=2)
        self.lbl_mov_det = tk.Label(grid_detalles, text="—", font=("Segoe UI", 10, "bold"), bg="#1e293b", fg="#60a5fa")
        self.lbl_mov_det.grid(row=1, column=1, sticky="w", padx=6, pady=2)

        tk.Label(grid_detalles, text="Distancia:", font=("Segoe UI", 9), bg="#1e293b", fg="#94a3b8").grid(row=2, column=0, sticky="w", pady=2)
        self.lbl_dist_det = tk.Label(grid_detalles, text="—", font=("Segoe UI", 10, "bold"), bg="#1e293b", fg="#ffffff")
        self.lbl_dist_det.grid(row=2, column=1, sticky="w", padx=6, pady=2)

        # Tarjeta 2: Registro de Auditoría (CSV)
        card_audit = tk.Frame(col_derecha, bg="#1e293b", bd=1, relief="solid", padx=10, pady=10)
        card_audit.pack(fill=tk.BOTH, expand=True)

        header_audit = tk.Frame(card_audit, bg="#1e293b")
        header_audit.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            header_audit,
            text="HISTORIAL DE AUDITORÍA RECIENTE",
            font=("Segoe UI", 9, "bold"),
            bg="#1e293b",
            fg="#94a3b8",
        ).pack(side=tk.LEFT)

        btn_refrescar_csv = tk.Button(
            header_audit,
            text="Refrescar",
            font=("Segoe UI", 8),
            bg="#334155",
            fg="#f8fafc",
            relief="flat",
            padx=6,
            pady=2,
            cursor="hand2",
            command=self._cargar_historial_csv,
        )
        btn_refrescar_csv.pack(side=tk.RIGHT)

        # Botón inferior para abrir el archivo CSV externo (siempre visible abajo)
        btn_abrir_csv = tk.Button(
            card_audit,
            text="Abrir archivo CSV (registro_accesos.csv)",
            font=("Segoe UI", 9),
            bg="#334155",
            fg="#f8fafc",
            activebackground="#475569",
            activeforeground="#ffffff",
            relief="flat",
            pady=5,
            cursor="hand2",
            command=self._abrir_archivo_csv,
        )
        btn_abrir_csv.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))

        # Tabla Treeview
        tree_frame = tk.Frame(card_audit, bg="#1e293b")
        tree_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        columnas = ("fecha", "hora", "residente", "movimiento")
        self.tree_auditoria = ttk.Treeview(
            tree_frame,
            columns=columnas,
            show="headings",
            selectmode="browse",
            height=7,
        )
        self.tree_auditoria.heading("fecha", text="Fecha")
        self.tree_auditoria.heading("hora", text="Hora")
        self.tree_auditoria.heading("residente", text="Residente")
        self.tree_auditoria.heading("movimiento", text="Movimiento")

        self.tree_auditoria.column("fecha", width=78, anchor="center")
        self.tree_auditoria.column("hora", width=68, anchor="center")
        self.tree_auditoria.column("residente", width=130, anchor="w")
        self.tree_auditoria.column("movimiento", width=90, anchor="center")

        scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree_auditoria.yview)
        self.tree_auditoria.configure(yscrollcommand=scroll_y.set)

        self.tree_auditoria.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree_auditoria.tag_configure("ENTRADA", foreground="#34d399")
        self.tree_auditoria.tag_configure("SALIDA", foreground="#60a5fa")

        # Cargar historial CSV existente
        self._cargar_historial_csv()

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2: COMPARACIÓN FACIAL 1:1
    # ──────────────────────────────────────────────────────────────────────────
    def _construir_tab_comparacion(self):
        panel_comp = tk.Frame(self.tab_comparacion, bg="#0f172a")
        panel_comp.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        # Fila superior: Dos tarjetas lado a lado
        fila_fotos = tk.Frame(panel_comp, bg="#0f172a")
        fila_fotos.pack(fill=tk.BOTH, expand=True)

        # Tarjeta Imagen Referencia A
        self.card_a = self._crear_tarjeta_imagen(
            fila_fotos,
            titulo="IMAGEN A — REFERENCIA",
            lado="A",
            preset_default="FotoA.jpg",
        )
        self.card_a.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        # Tarjeta Imagen Prueba B
        self.card_b = self._crear_tarjeta_imagen(
            fila_fotos,
            titulo="IMAGEN B — PRUEBA",
            lado="B",
            preset_default="FotoB.jpg",
        )
        self.card_b.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))

        # Fila de Controles y Resultados
        card_control_comp = tk.Frame(panel_comp, bg="#1e293b", bd=1, relief="solid", padx=15, pady=12)
        card_control_comp.pack(fill=tk.X, pady=(10, 0))

        # Controles
        fila_ctrl = tk.Frame(card_control_comp, bg="#1e293b")
        fila_ctrl.pack(fill=tk.X)

        tk.Label(
            fila_ctrl,
            text="Umbral de Distancia Máxima:",
            font=("Segoe UI", 10, "bold"),
            bg="#1e293b",
            fg="#f8fafc",
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.var_umbral_1a1 = tk.DoubleVar(value=UMBRAL_DISTANCIA)
        slider_1a1 = ttk.Scale(
            fila_ctrl,
            from_=0.20,
            to=0.90,
            variable=self.var_umbral_1a1,
            orient=tk.HORIZONTAL,
            length=180,
            command=lambda v: self.lbl_val_umbral_1a1.config(text=f"{float(v):.2f}"),
        )
        slider_1a1.pack(side=tk.LEFT, padx=(0, 8))

        self.lbl_val_umbral_1a1 = tk.Label(
            fila_ctrl,
            text=f"{UMBRAL_DISTANCIA:.2f}",
            font=("Segoe UI", 10, "bold"),
            bg="#1e293b",
            fg="#60a5fa",
            width=5,
        )
        self.lbl_val_umbral_1a1.pack(side=tk.LEFT, padx=(0, 25))

        self.btn_comparar_1a1 = tk.Button(
            fila_ctrl,
            text="Comparar Rostros 1:1",
            font=("Segoe UI", 11, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            padx=20,
            pady=7,
            cursor="hand2",
            command=self._ejecutar_comparacion_1a1,
        )
        self.btn_comparar_1a1.pack(side=tk.LEFT)

        # Banner de Resultado 1:1
        self.banner_resultado_1a1 = tk.Label(
            card_control_comp,
            text="Seleccione las dos imágenes y presione 'Comparar Rostros 1:1'",
            font=("Segoe UI", 12, "bold"),
            bg="#334155",
            fg="#f8fafc",
            pady=10,
        )
        self.banner_resultado_1a1.pack(fill=tk.X, pady=(12, 6))

        # Detalle de la métrica
        self.lbl_metrica_1a1 = tk.Label(
            card_control_comp,
            text="Distancia: — | Menor a umbral indica coincidencia (misma identidad)",
            font=("Segoe UI", 9),
            bg="#1e293b",
            fg="#94a3b8",
        )
        self.lbl_metrica_1a1.pack()

        # Cargar imágenes por defecto si existen
        self.root.after(200, self._cargar_presets_iniciales_1a1)

    def _crear_tarjeta_imagen(self, padre, titulo: str, lado: str, preset_default: str):
        card = tk.Frame(padre, bg="#1e293b", bd=1, relief="solid", padx=10, pady=8)

        # Header de la tarjeta
        header = tk.Frame(card, bg="#1e293b")
        header.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            header,
            text=titulo,
            font=("Segoe UI", 10, "bold"),
            bg="#1e293b",
            fg="#60a5fa" if lado == "A" else "#34d399",
        ).pack(side=tk.LEFT)

        lbl_estado_foto = tk.Label(
            header,
            text="Sin imagen",
            font=("Segoe UI", 8),
            bg="#334155",
            fg="#94a3b8",
            padx=6,
            pady=1,
        )
        lbl_estado_foto.pack(side=tk.RIGHT)

        if lado == "A":
            self.lbl_estado_foto_a = lbl_estado_foto
        else:
            self.lbl_estado_foto_b = lbl_estado_foto

        # Visor de la foto
        canvas_foto = tk.Canvas(card, bg="#020617", highlightthickness=0, height=270)
        canvas_foto.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        if lado == "A":
            self.canvas_foto_a = canvas_foto
        else:
            self.canvas_foto_b = canvas_foto

        # Barra de selección
        barra_sel = tk.Frame(card, bg="#1e293b")
        barra_sel.pack(fill=tk.X)

        btn_examinar = tk.Button(
            barra_sel,
            text=f"Seleccionar Imagen {lado}...",
            font=("Segoe UI", 9, "bold"),
            bg="#334155",
            fg="#f8fafc",
            activebackground="#475569",
            activeforeground="#ffffff",
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
            command=lambda: self._examinar_imagen(lado),
        )
        btn_examinar.pack(side=tk.LEFT, padx=(0, 8))

        # Botones Presets
        tk.Label(barra_sel, text="Presets:", font=("Segoe UI", 8), bg="#1e293b", fg="#94a3b8").pack(side=tk.LEFT, padx=(0, 4))
        for p in ("FotoA.jpg", "FotoB.jpg", "FotoC.jpg"):
            btn_p = tk.Button(
                barra_sel,
                text=p.replace(".jpg", ""),
                font=("Segoe UI", 8),
                bg="#1e293b",
                fg="#60a5fa",
                activebackground="#334155",
                activeforeground="#ffffff",
                relief="groove",
                padx=5,
                pady=1,
                cursor="hand2",
                command=lambda ruta=p: self._cargar_imagen_1a1(lado, ruta),
            )
            btn_p.pack(side=tk.LEFT, padx=2)

        lbl_nombre_archivo = tk.Label(
            card,
            text="Ningún archivo seleccionado",
            font=("Segoe UI", 8),
            bg="#1e293b",
            fg="#64748b",
            anchor="w",
        )
        lbl_nombre_archivo.pack(fill=tk.X, pady=(4, 0))

        if lado == "A":
            self.lbl_archivo_a = lbl_nombre_archivo
        else:
            self.lbl_archivo_b = lbl_nombre_archivo

        return card

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODOS DE LA BASE DE DATOS DE RESIDENTES
    # ──────────────────────────────────────────────────────────────────────────
    def _iniciar_carga_residentes(self):
        hilo = threading.Thread(target=self._tarea_cargar_residentes, daemon=True)
        hilo.start()

    def _tarea_cargar_residentes(self):
        try:
            self.root.after(0, lambda: self.lbl_estado_bd.config(text="Cargando residentes...", fg="#fbbf24"))

            cache_dir = os.path.join(DIRECTORIO_BASE, "data")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, "biometria_cache.pkl")

            cache = {}
            if os.path.isfile(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        cache = pickle.load(f)
                except Exception:
                    cache = {}

            archivos_residentes = sorted([
                f for f in os.listdir(DIRECTORIO_RESIDENTES)
                if f.lower().endswith(EXTENSIONES_IMAGEN_VALIDAS)
            ])

            cache_mtimes = cache.get("mtimes", {})
            cache_encodings = cache.get("encodings_map", {})

            necesita_guardar = False
            encodings_actuales = {}

            for archivo in archivos_residentes:
                ruta = os.path.join(DIRECTORIO_RESIDENTES, archivo)
                mtime = os.path.getmtime(ruta)
                if archivo in cache_encodings and cache_mtimes.get(archivo) == mtime:
                    encodings_actuales[archivo] = cache_encodings[archivo]
                else:
                    necesita_guardar = True
                    with dlib_lock:
                        img_rgb = cargar_imagen_segura_rgb(ruta)
                        if img_rgb is not None:
                            locs = fr.face_locations(img_rgb)
                            if locs:
                                enc = fr.face_encodings(img_rgb, locs)[0]
                                nombre_res = os.path.splitext(archivo)[0]
                                encodings_actuales[archivo] = (nombre_res, enc)

            if necesita_guardar or len(cache_mtimes) != len(archivos_residentes):
                nuevo_cache = {
                    "encodings_map": encodings_actuales,
                    "mtimes": {
                        f: os.path.getmtime(os.path.join(DIRECTORIO_RESIDENTES, f))
                        for f in archivos_residentes
                    },
                }
                try:
                    with open(cache_path, "wb") as f:
                        pickle.dump(nuevo_cache, f)
                except Exception:
                    pass

            self.nombres_validos = [val[0] for val in encodings_actuales.values()]
            self.lista_codificada = [val[1] for val in encodings_actuales.values()]

            def actualizar_ui():
                total = len(self.nombres_validos)
                self.lbl_estado_bd.config(
                    text=f"{total} Residentes Activos",
                    fg="#34d399",
                )
                # Iniciar la cámara web automáticamente al completar la carga
                if not self.camara_activa:
                    self.root.after(300, self._iniciar_camara)

            self.root.after(0, actualizar_ui)
        except Exception as e:
            def mostrar_error():
                self.lbl_estado_bd.config(text="Error al cargar residentes", fg="#ef4444")
                messagebox.showerror("Error de Base de Datos", str(e))

            self.root.after(0, mostrar_error)

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODOS DE CÁMARA WEB (TAB 1)
    # ──────────────────────────────────────────────────────────────────────────
    def _dibujar_placeholder_camara(self, texto: str):
        self.canvas_video.delete("all")
        w = self.canvas_video.winfo_width()
        h = self.canvas_video.winfo_height()
        if w <= 10 or h <= 10:
            w = 640
            h = 420
        self.canvas_video.create_rectangle(0, 0, w, h, fill="#020617", outline="")
        self.canvas_video.create_text(
            w // 2,
            h // 2 - 25,
            text="[CAM]",
            font=("Segoe UI", 32, "bold"),
            fill="#334155",
        )
        self.canvas_video.create_text(
            w // 2,
            h // 2 + 25,
            text=texto,
            font=("Segoe UI", 11),
            fill="#64748b",
            justify="center",
        )

    def _al_redimensionar_canvas(self):
        if not self.camara_activa:
            self._dibujar_placeholder_camara("Cámara inactiva\nPresione 'Iniciar Cámara' para comenzar")

    def _toggle_camara(self):
        if self.camara_activa:
            self._detener_camara()
        else:
            self._iniciar_camara()

    def _iniciar_camara(self):
        if not self.lista_codificada:
            messagebox.showwarning(
                "Base no lista",
                "Aguarde a que finalice la carga de los residentes antes de activar la cámara.",
            )
            return

        indice = self.var_cam_index.get()
        self.captura = car.abrir_camara(indice)

        if not self.captura.isOpened():
            messagebox.showerror(
                "Error de Cámara",
                f"No se pudo acceder al dispositivo de video (Cámara {indice}).",
            )
            return

        self.camara_activa = True
        self.btn_toggle_cam.config(
            text="⏹ Detener Cámara",
            bg="#ef4444",
            activebackground="#dc2626",
        )
        self.badge_estado.config(
            text="MONITOREANDO EN VIVO...",
            bg="#1e3a8a",
            fg="#93c5fd",
        )
        self._loop_camara()

    def _detener_camara(self):
        self.camara_activa = False
        if self.captura is not None:
            try:
                self.captura.release()
            except Exception:
                pass
            self.captura = None

        self.btn_toggle_cam.config(
            text="▶ Iniciar Cámara",
            bg="#10b981",
            activebackground="#059669",
        )
        self.badge_estado.config(
            text="CÁMARA INACTIVA",
            bg="#334155",
            fg="#94a3b8",
        )
        self._dibujar_placeholder_camara("Cámara detenida\nPresione 'Iniciar Cámara' para reanudar")

    def _loop_camara(self):
        if not self.camara_activa or self.captura is None:
            return

        exito, frame = self.captura.read()
        if not exito or frame is None:
            self.root.after(30, self._loop_camara)
            return

        self.frame_count += 1

        # Ejecutar reconocimiento biométrico cada 3 fotogramas con reducción 0.5x
        if self.frame_count % 3 == 0:
            self._procesar_reconocimiento_fotograma(frame)

        # Dibujar las detecciones sobre el fotograma actual
        self._dibujar_overlays_fotograma(frame)

        # Mostrar fotograma en el Canvas
        self._mostrar_frame_en_canvas(frame)

        # Programar siguiente ciclo (~30 FPS)
        self.root.after(25, self._loop_camara)

    def _procesar_reconocimiento_fotograma(self, frame: np.ndarray):
        umbral = self.var_umbral_cam.get()

        # Escalar a la mitad para mayor fluidez
        pequeno = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_pequeno = cv2.cvtColor(pequeno, cv2.COLOR_BGR2RGB)

        with dlib_lock:
            caras_ubicaciones = fr.face_locations(rgb_pequeno)
            caras_codificaciones = fr.face_encodings(rgb_pequeno, caras_ubicaciones)

        self.ultimas_caras = []
        self.ultimas_etiquetas = []

        if not caras_ubicaciones:
            self.badge_estado.config(
                text="MONITOREANDO (SIN ROSTROS)",
                bg="#1e293b",
                fg="#64748b",
            )
            return

        for codif, ubic in zip(caras_codificaciones, caras_ubicaciones):
            # Escalar coordenadas de regreso al tamaño original
            arriba, derecha, abajo, izquierda = [v * 2 for v in ubic]
            self.ultimas_caras.append((arriba, derecha, abajo, izquierda))

            distancias = fr.face_distance(self.lista_codificada, codif)
            indice_min = int(np.argmin(distancias))
            distancia_min = float(distancias[indice_min])

            if distancia_min <= umbral:
                nombre = self.nombres_validos[indice_min]
                ahora_ts = time.time()
                ultimo_ts = self.ultimos_registros.get(nombre, 0.0)

                if (ahora_ts - ultimo_ts) >= TIEMPO_ENTRE_REGISTROS_SEG:
                    movimiento = car.registrar_acceso_auditoria(nombre)
                    self.ultimos_registros[nombre] = ahora_ts
                    self._cargar_historial_csv()
                else:
                    movimiento = car.determinar_siguiente_movimiento(nombre, ARCHIVO_REGISTRO_CSV)
                    movimiento = "SALIDA" if movimiento == "ENTRADA" else "ENTRADA"

                self.ultimas_etiquetas.append((f"{nombre} ({movimiento})", True, distancia_min))

                # Actualizar tarjeta de estado visual
                self.badge_estado.config(
                    text=f"ACCESO AUTORIZADO — {movimiento}",
                    bg="#065f46",
                    fg="#34d399",
                )
                self.lbl_nombre_det.config(text=nombre, fg="#34d399")
                self.lbl_mov_det.config(text=movimiento, fg="#60a5fa")
                self.lbl_dist_det.config(
                    text=f"{distancia_min:.3f} (Umbral: {umbral:.2f})",
                    fg="#ffffff",
                )
            else:
                self.ultimas_etiquetas.append(("ACCESO DENEGADO", False, distancia_min))
                self.badge_estado.config(
                    text="ACCESO DENEGADO (NO REGISTRADO)",
                    bg="#7f1d1d",
                    fg="#f87171",
                )
                self.lbl_nombre_det.config(text="Desconocido", fg="#f87171")
                self.lbl_mov_det.config(text="Bloqueado", fg="#f87171")
                self.lbl_dist_det.config(
                    text=f"{distancia_min:.3f} > {umbral:.2f}",
                    fg="#f87171",
                )

    def _dibujar_overlays_fotograma(self, frame: np.ndarray):
        for ubic, (etiqueta, autorizado, dist) in zip(self.ultimas_caras, self.ultimas_etiquetas):
            arriba, derecha, abajo, izquierda = ubic
            color = (0, 200, 0) if autorizado else (0, 0, 220)

            # Recuadro alrededor de la cara
            cv2.rectangle(frame, (izquierda, arriba), (derecha, abajo), color, 2)

            # Fondo para el texto
            cv2.rectangle(
                frame,
                (izquierda, abajo - 32),
                (derecha, abajo),
                color,
                cv2.FILLED,
            )

            # Texto con nombre / estado
            cv2.putText(
                frame,
                etiqueta,
                (izquierda + 6, abajo - 8),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _mostrar_frame_en_canvas(self, frame: np.ndarray):
        cw = self.canvas_video.winfo_width()
        ch = self.canvas_video.winfo_height()

        if cw <= 1 or ch <= 1:
            cw, ch = 640, 480

        h, w = frame.shape[:2]
        factor = min(cw / w, ch / h)
        nw, nh = max(1, int(w * factor)), max(1, int(h * factor))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb).resize((nw, nh), Image.Resampling.BILINEAR)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.canvas_video.delete("all")
        self.canvas_video.create_image(cw // 2, ch // 2, image=img_tk)
        self.canvas_video._img_ref = img_tk

    def _ejecutar_captura_unica(self):
        if not self.lista_codificada:
            messagebox.showwarning("Aviso", "Cargando residentes, aguarde un instante.")
            return

        indice = self.var_cam_index.get()
        captura = car.abrir_camara(indice)
        exito, frame = captura.read()
        captura.release()

        if not exito or frame is None:
            messagebox.showerror("Error", "No se pudo obtener la captura de la cámara.")
            return

        self._procesar_reconocimiento_fotograma(frame)
        self._dibujar_overlays_fotograma(frame)
        self._mostrar_frame_en_canvas(frame)

    def _cargar_historial_csv(self):
        for item in self.tree_auditoria.get_children():
            self.tree_auditoria.delete(item)

        if not os.path.isfile(ARCHIVO_REGISTRO_CSV):
            car.inicializar_archivo_auditoria(ARCHIVO_REGISTRO_CSV)
            return

        try:
            with open(ARCHIVO_REGISTRO_CSV, mode="r", encoding="utf-8") as f:
                lineas = f.readlines()

            for linea in reversed(lineas[1:]):  # Mostrar los más recientes arriba
                partes = [p.strip() for p in linea.strip().split(",")]
                if len(partes) >= 4:
                    residente = partes[0]
                    fecha = partes[1]
                    hora = partes[2]
                    movimiento = partes[3]
                    tag = "ENTRADA" if movimiento == "ENTRADA" else "SALIDA"
                    self.tree_auditoria.insert(
                        "",
                        tk.END,
                        values=(fecha, hora, residente, movimiento),
                        tags=(tag,),
                    )
        except Exception as e:
            print(f"[ERROR] No se pudo leer el CSV: {e}")

    def _abrir_archivo_csv(self):
        if not os.path.isfile(ARCHIVO_REGISTRO_CSV):
            car.inicializar_archivo_auditoria(ARCHIVO_REGISTRO_CSV)
        try:
            os.startfile(ARCHIVO_REGISTRO_CSV)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir el archivo CSV: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODOS DE COMPARACIÓN FACIAL 1:1 (TAB 2)
    # ──────────────────────────────────────────────────────────────────────────
    def _cargar_presets_iniciales_1a1(self):
        foto_a = os.path.join(DIRECTORIO_BASE, "FotoA.jpg")
        foto_b = os.path.join(DIRECTORIO_BASE, "FotoB.jpg")
        if os.path.isfile(foto_a):
            self._cargar_imagen_1a1("A", foto_a)
        if os.path.isfile(foto_b):
            self._cargar_imagen_1a1("B", foto_b)

    def _examinar_imagen(self, lado: str):
        tipos = [
            ("Imágenes admitidas", "*.jpg *.jpeg *.png *.bmp"),
            ("Todos los archivos", "*.*"),
        ]
        ruta = filedialog.askopenfilename(
            title=f"Seleccionar Imagen {lado}",
            initialdir=DIRECTORIO_BASE,
            filetypes=tipos,
        )
        if ruta:
            self._cargar_imagen_1a1(lado, ruta)

    def _cargar_imagen_1a1(self, lado: str, ruta: str):
        if not os.path.isabs(ruta):
            ruta = os.path.join(DIRECTORIO_BASE, ruta)

        if not os.path.isfile(ruta):
            messagebox.showerror("Error", f"Archivo no encontrado: {ruta}")
            return

        nombre_archivo = os.path.basename(ruta)
        img_rgb = cargar_imagen_segura_rgb(ruta)
        if img_rgb is None:
            messagebox.showerror("Error", f"No se pudo decodificar la imagen: {nombre_archivo}")
            return

        # Detectar rostro y codificar
        with dlib_lock:
            ubicaciones = motor.detectar_rostros(img_rgb)
            caja = ubicaciones[0] if len(ubicaciones) == 1 else None
            encoding = motor.codificar_rostro(img_rgb, caja) if caja is not None else None

        if lado == "A":
            self.ruta_img_a = ruta
            self.img_rgb_a = img_rgb
            self.caja_a = caja
            self.encoding_a = encoding
            self.lbl_archivo_a.config(text=nombre_archivo)
            self._actualizar_badge_foto("A", len(ubicaciones))
            self._dibujar_foto_en_canvas("A", img_rgb, caja)
        else:
            self.ruta_img_b = ruta
            self.img_rgb_b = img_rgb
            self.caja_b = caja
            self.encoding_b = encoding
            self.lbl_archivo_b.config(text=nombre_archivo)
            self._actualizar_badge_foto("B", len(ubicaciones))
            self._dibujar_foto_en_canvas("B", img_rgb, caja)

    def _actualizar_badge_foto(self, lado: str, num_rostros: int):
        lbl = self.lbl_estado_foto_a if lado == "A" else self.lbl_estado_foto_b
        if num_rostros == 1:
            lbl.config(text="✓ 1 rostro detectado", bg="#065f46", fg="#34d399")
        elif num_rostros == 0:
            lbl.config(text="⚠ 0 rostros detectados", bg="#7f1d1d", fg="#fca5a5")
        else:
            lbl.config(text=f"⚠ {num_rostros} rostros detectados", bg="#78350f", fg="#fcd34d")

    def _dibujar_foto_en_canvas(
        self,
        lado: str,
        img_rgb: np.ndarray,
        caja: Optional[Tuple[int, int, int, int]],
        color_box=(0, 220, 0),
    ):
        canvas = self.canvas_foto_a if lado == "A" else self.canvas_foto_b
        cw = canvas.winfo_width() or 380
        ch = canvas.winfo_height() or 270

        copia = img_rgb.copy()
        if caja is not None:
            arriba, derecha, abajo, izquierda = caja
            cv2.rectangle(copia, (izquierda, arriba), (derecha, abajo), color_box, 4)

        h, w = copia.shape[:2]
        factor = min(cw / w, ch / h)
        nw, nh = max(1, int(w * factor)), max(1, int(h * factor))

        img_pil = Image.fromarray(copia).resize((nw, nh), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=img_tk)
        if lado == "A":
            canvas._img_a_ref = img_tk
        else:
            canvas._img_b_ref = img_tk

    def _ejecutar_comparacion_1a1(self):
        if self.encoding_a is None or self.caja_a is None:
            messagebox.showwarning(
                "Imagen A Inválida",
                "La Imagen A debe contener exactamente un rostro reconocible.",
            )
            return

        if self.encoding_b is None or self.caja_b is None:
            messagebox.showwarning(
                "Imagen B Inválida",
                "La Imagen B debe contener exactamente un rostro reconocible.",
            )
            return

        umbral = self.var_umbral_1a1.get()
        distancia = float(fr.face_distance([self.encoding_a], self.encoding_b)[0])
        coincide = distancia <= umbral

        color_box = (0, 220, 0) if coincide else (220, 0, 0)
        self._dibujar_foto_en_canvas("A", self.img_rgb_a, self.caja_a, color_box)
        self._dibujar_foto_en_canvas("B", self.img_rgb_b, self.caja_b, color_box)

        if coincide:
            self.banner_resultado_1a1.config(
                text="[COINCIDENCIA CONFIRMADA] — MISMA IDENTIDAD",
                bg="#065f46",
                fg="#34d399",
            )
        else:
            self.banner_resultado_1a1.config(
                text="[SIN COINCIDENCIA] — IDENTIDADES DISTINTAS",
                bg="#7f1d1d",
                fg="#fca5a5",
            )

        self.lbl_metrica_1a1.config(
            text=(
                f"Distancia biométrica calculada: {distancia:.4f}  |  "
                f"Umbral máximo permitido: {umbral:.2f}  |  "
                f"{'Menor al umbral (Válido)' if coincide else 'Superior al umbral (Denegado)'}"
            ),
            fg="#f8fafc",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # CIERRE DE APLICACIÓN
    # ──────────────────────────────────────────────────────────────────────────
    def _al_cerrar(self):
        self.camara_activa = False
        if self.captura is not None:
            try:
                self.captura.release()
            except Exception:
                pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = AppControlAccesoGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
