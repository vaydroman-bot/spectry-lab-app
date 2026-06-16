import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates
import anthropic
import base64
import hashlib
import io
import os

# ─── CONFIGURACIÓN ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Spectry Lab", layout="wide", page_icon="🧪")

# El esquema de color base (fondo claro, tarjetas blancas, texto oscuro) se
# define en .streamlit/config.toml para que todos los widgets nativos de
# Streamlit (botones, sliders, alerts...) hereden el tema automáticamente.
# Aquí solo se añaden los componentes decorativos que Streamlit no ofrece.
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1180px; }

  div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px !important;
    box-shadow: 0 1px 3px rgba(16,24,40,0.06), 0 1px 2px rgba(16,24,40,0.04);
  }

  .stButton > button[kind="primary"] {
    background: linear-gradient(90deg,#0EA5E9,#6366F1);
    border: none;
  }

  /* Tarjeta de resultado autocontenida */
  .card {
    background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 16px;
    padding: 24px; box-shadow: 0 1px 3px rgba(16,24,40,0.06), 0 1px 2px rgba(16,24,40,0.04);
    margin: 12px 0;
  }

  /* Cabecera de pasos */
  .steps-row { display:flex; justify-content:center; gap:10px; margin: 4px 0 22px 0; flex-wrap: wrap; }
  .step-pill {
    display:flex; align-items:center; gap:8px; background:#fff; border:1px solid #E4E7EC;
    border-radius:999px; padding:7px 16px; font-size:.82rem; font-weight:600; color:#8A94A6;
  }
  .step-pill.active { border-color:#0EA5E9; color:#0EA5E9; background:#EFF8FF; }
  .step-pill.done   { border-color:#16A34A; color:#16A34A; background:#F0FDF4; }
  .step-num {
    width:20px; height:20px; border-radius:50%; background:#D0D5DD; color:#fff;
    display:flex; align-items:center; justify-content:center; font-size:.7rem; flex-shrink:0;
  }
  .step-pill.active .step-num { background:#0EA5E9; }
  .step-pill.done .step-num   { background:#16A34A; }

  /* Gauge circular de similitud */
  .gauge-wrap { display:flex; justify-content:center; padding: 4px 0; }
  .gauge {
    width:168px; height:168px; border-radius:50%;
    background: conic-gradient(var(--gauge-color) calc(var(--gauge-deg) * 1deg), #E4E7EC 0deg);
    display:flex; align-items:center; justify-content:center;
  }
  .gauge-inner {
    width:134px; height:134px; border-radius:50%; background:#fff;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    box-shadow: inset 0 0 0 1px #E4E7EC;
  }
  .gauge-pct   { font-size:2.1rem; font-weight:800; color:#101828; line-height:1; }
  .gauge-label { font-size:.68rem; color:#8A94A6; letter-spacing:1.5px; margin-top:3px; }

  /* Comparación VS de muestras */
  .vs-row { display:flex; align-items:center; gap:18px; justify-content:center; }
  .vs-swatch-box { text-align:center; }
  .vs-swatch {
    width:84px; height:84px; border-radius:12px; border:1px solid #E4E7EC;
    box-shadow:0 1px 2px rgba(16,24,40,.08);
  }
  .vs-label { font-size:.72rem; color:#344054; font-weight:600; margin-top:6px; }
  .vs-hex   { font-size:.72rem; color:#8A94A6; font-family: monospace; }
  .vs-versus { font-size:1.05rem; font-weight:800; color:#D0D5DD; }

  /* Insignias de calidad */
  .badge {
    display:inline-flex; align-items:center; gap:4px; padding:3px 11px;
    border-radius:999px; font-size:.74rem; font-weight:600; margin: 2px 0;
  }
  .badge-ok  { background:#F0FDF4; color:#16A34A; }
  .badge-mid { background:#FFFBEB; color:#D97706; }
  .badge-bad { background:#FEF2F2; color:#DC2626; }

  .dE-ok  { color:#16A34A; }
  .dE-mid { color:#D97706; }
  .dE-bad { color:#DC2626; }

  .chip {
    display:inline-block; width:22px; height:22px; border-radius:6px;
    border:1px solid #E4E7EC; margin:2px; vertical-align:middle;
  }
  .hint { color:#8A94A6; font-size:.8rem; text-align:center; margin-bottom:4px; }
</style>
""", unsafe_allow_html=True)

# ─── SISTEMAS DE TINTES ───────────────────────────────────────────────────────
SISTEMAS = {
    "BESA Urki-Mix": {
        "negro": "9005", "blanco": "Blanco Base",
        "aluminio": "Aluminio", "perla": "Perla Blanca",
        "ocre": "Ocre Amarillo", "rojo": "Rojo",
        "azul": "Azul", "verde": "Verde", "violeta": "Violeta",
    },
    "NOVOL Spectral": {
        "negro": "SB-1000", "blanco": "SB-2000",
        "aluminio": "B-810", "perla": "B-Perla",
        "ocre": "B-Ocre", "rojo": "B-Rojo",
        "azul": "B-Azul", "verde": "B-Verde", "violeta": "B-Violeta",
    },
    "Standox Standohyd": {
        "negro": "W001", "blanco": "W040",
        "aluminio": "W075", "perla": "W090",
        "ocre": "W064", "rojo": "W036",
        "azul": "W051", "verde": "W057", "violeta": "W058",
    },
}

# ─── ESTADO DE SESIÓN ─────────────────────────────────────────────────────────
ESTADO_POR_DEFECTO = {
    "puntos_coche": [], "puntos_muestra": [],
    "rgb_coche": [],    "rgb_muestra": [],
    "std_coche": [],    "std_muestra": [],
    "prev_coche": None, "prev_muestra": None,
    "hash_coche": None, "hash_muestra": None,
    "imgfull_coche": None, "imgfull_muestra": None,
    "b64_coche": None,  "b64_muestra": None,
    "factor_coche": None, "factor_muestra": None,
    "calidad_coche": None, "calidad_muestra": None,
    "ilu_coche": None, "ilu_muestra": None,
    "resultado_ia": None,
}
for k, v in ESTADO_POR_DEFECTO.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── HELPERS DE IMAGEN ───────────────────────────────────────────────────────
def cargar_imagen(f):
    f.seek(0)
    img = Image.open(f)
    img = ImageOps.exif_transpose(img)  # corrige fotos de móvil giradas por el sensor de orientación
    return img.convert("RGB")

def hash_archivo(f):
    f.seek(0)
    h = hashlib.md5(f.read()).hexdigest()
    f.seek(0)
    return h

def redimensionar(img, lado_max):
    w, h = img.size
    lado = max(w, h)
    if lado <= lado_max:
        return img.copy()
    r = lado_max / lado
    return img.resize((max(1, int(w * r)), max(1, int(h * r))), Image.Resampling.LANCZOS)

def dibujar_miras(img, puntos):
    arr = np.array(img.copy())
    for i, (x, y) in enumerate(puntos):
        for col, g in [((0, 0, 0), 2), ((0, 255, 0), 1)]:
            cv2.line(arr, (x - 15, y), (x + 15, y), col, g)
            cv2.line(arr, (x, y - 15), (x, y + 15), col, g)
            cv2.circle(arr, (x, y), 10, col, g)
        cv2.putText(arr, str(i + 1), (x + 12, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(arr, str(i + 1), (x + 12, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return Image.fromarray(arr)

def muestrear_color(img, x, y, radio):
    arr = np.array(img)
    h, w = arr.shape[:2]
    parche = arr[max(0, y - radio):min(h, y + radio + 1),
                 max(0, x - radio):min(w, x + radio + 1)]
    pixeles = parche.reshape(-1, 3).astype(np.float32)
    rgb = tuple(np.median(pixeles, axis=0).round().astype(int))
    # Desviación por canal (no agrupada): un color saturado uniforme (R≠G≠B)
    # no debe contar como "ruido" — solo la variación espacial dentro de cada
    # canal indica falta de uniformidad real (reflejos, textura, sombra).
    std = float(np.mean(np.std(pixeles, axis=0)))
    return rgb, std

def zoom_punto(img, x, y, radio_zoom=70, tam=220):
    arr = np.array(img)
    h, w = arr.shape[:2]
    y1, y2 = max(0, y - radio_zoom), min(h, y + radio_zoom)
    x1, x2 = max(0, x - radio_zoom), min(w, x + radio_zoom)
    crop = Image.fromarray(arr[y1:y2, x1:x2])
    return crop.resize((tam, tam), Image.Resampling.LANCZOS)

def a_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()

def hex_color(rgb):
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"

# ─── CALIDAD DE CAPTURA ───────────────────────────────────────────────────────
def analizar_calidad_captura(img):
    """Detecta automáticamente desenfoque y sobre/sub-exposición de la foto recién subida."""
    arr = np.array(img)
    gris = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())
    sobreexp = float(np.mean(np.all(arr >= 248, axis=2))) * 100
    subexp = float(np.mean(np.all(arr <= 10, axis=2))) * 100

    avisos = []
    if nitidez < 60:
        avisos.append(("bad", f"Imagen borrosa (nitidez {nitidez:.0f}) — apoya el móvil y espera a que enfoque antes de disparar"))
    elif nitidez < 120:
        avisos.append(("mid", f"Nitidez algo baja ({nitidez:.0f}) — si puedes, repite la foto"))
    if sobreexp > 3:
        avisos.append(("bad", f"Zonas quemadas ({sobreexp:.1f}%) — evita el flash y el sol directo sobre la pintura"))
    if subexp > 15:
        avisos.append(("mid", f"Zonas muy oscuras ({subexp:.1f}%) — busca un punto con más luz"))
    return {"nitidez": nitidez, "sobreexp": sobreexp, "subexp": subexp, "avisos": avisos}

# ─── ILUMINANTE Y CALIBRACIÓN SIN HARDWARE ESPECÍFICO ────────────────────────
def estimar_iluminante(img):
    """Estimación tipo 'white-patch': promedia el 10% de píxeles más brillantes
    y no quemados como proxy del color de la luz de la escena. Es más fiable que
    un gray-world clásico cuando el objeto fotografiado es un único color saturado
    (el gray-world trataría ese tono como un error a corregir)."""
    arr = np.array(img).reshape(-1, 3).astype(np.float32)
    brillo = arr.sum(axis=1)
    no_quemado = np.all(arr < 250, axis=1)
    base = arr[no_quemado] if no_quemado.any() else arr
    brillo_base = brillo[no_quemado] if no_quemado.any() else brillo
    umbral = np.percentile(brillo_base, 90)
    seleccion = base[brillo_base >= umbral]
    if len(seleccion) == 0:
        seleccion = base
    return seleccion.mean(axis=0)

def diferencia_iluminantes(ilu_a, ilu_b):
    va = ilu_a / (np.linalg.norm(ilu_a) + 1e-6)
    vb = ilu_b / (np.linalg.norm(ilu_b) + 1e-6)
    return float(np.linalg.norm(va - vb))

def calcular_factor_calibracion(rgb_medido, objetivo=200.0):
    """Factor por canal para que un punto neutro (gris/blanco) tocado por el
    usuario quede en gris real. No requiere ninguna carta de color: sirve
    cualquier papel, pared o trapo neutro presente en la propia foto."""
    factor = np.array([objetivo / max(float(c), 1.0) for c in rgb_medido], dtype=np.float32)
    return np.clip(factor, 0.6, 1.8)

def aplicar_calibracion(rgb, factor):
    if factor is None:
        return tuple(int(c) for c in rgb)
    return tuple(int(round(min(255, max(0, c * f)))) for c, f in zip(rgb, factor))

# ─── COLORIMETRÍA ─────────────────────────────────────────────────────────────
def rgb_a_lab(rgb):
    # OpenCV COLOR_RGB2Lab con float32 [0,1]: L en [0,100], a/b en [0,255] → restar 128
    n = np.array([[[rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]]],
                 dtype=np.float32)
    L, a, b = cv2.cvtColor(n, cv2.COLOR_RGB2Lab)[0][0]
    return float(L), float(a - 128), float(b - 128)

def lab_promedio(lista_rgb):
    if not lista_rgb:
        return None
    return tuple(np.mean([rgb_a_lab(c) for c in lista_rgb], axis=0))

def rgb_promedio(lista_rgb):
    if not lista_rgb:
        return None
    return tuple(int(x) for x in np.round(np.mean(lista_rgb, axis=0)))

def interpretar_dE(dE):
    if dE < 1.0:
        return "Diferencia imperceptible al ojo humano", "ok"
    elif dE < 2.0:
        return "Solo detectable por expertos entrenados", "ok"
    elif dE < 3.5:
        return "Perceptible — umbral de aceptación industrial", "mid"
    elif dE < 5.0:
        return "Diferencia clara — ajuste necesario", "mid"
    elif dE < 10.0:
        return "Diferencia notable — ajuste importante", "bad"
    else:
        return "Colores muy distintos — valorar reformular", "bad"

def lista_calibrada(sfx):
    factor = st.session_state.get(f"factor_{sfx}")
    crudos = st.session_state.get(f"rgb_{sfx}", [])
    return [aplicar_calibracion(c, factor) for c in crudos]

# ─── AJUSTE RÁPIDO (sin IA) ──────────────────────────────────────────────────
def ajuste_rapido(sistema, dL, da, db):
    """
    dL = L_coche - L_prueba
      dL > 0 → coche más claro → prueba demasiado oscura → hay que ACLARAR la prueba
      dL < 0 → coche más oscuro → prueba demasiado clara → hay que OSCURECER la prueba
    da = a*_coche - a*_prueba
      da > 0 → falta rojo en la prueba → añadir rojo
      da < 0 → sobra rojo en la prueba → añadir verde
    db = b*_coche - b*_prueba
      db > 0 → falta amarillo en la prueba → añadir ocre
      db < 0 → sobra amarillo en la prueba → añadir azul
    """
    t = SISTEMAS[sistema]
    r = []
    if dL > 2:
        r.append(f"Prueba demasiado **oscura** (ΔL={dL:+.1f}) → aclarar con **{t['aluminio']}** o **{t['blanco']}**")
    elif dL < -2:
        r.append(f"Prueba demasiado **clara** (ΔL={dL:+.1f}) → oscurecer con **{t['negro']}**")
    if da > 1.5:
        r.append(f"Falta **rojo** (Δa*={da:+.1f}) → añadir **{t['rojo']}**")
    elif da < -1.5:
        r.append(f"Sobra **rojo** (Δa*={da:+.1f}) → añadir **{t['verde']}**")
    if db > 1.5:
        r.append(f"Falta **amarillo** (Δb*={db:+.1f}) → añadir **{t['ocre']}**")
    elif db < -1.5:
        r.append(f"Sobra **amarillo** (Δb*={db:+.1f}) → añadir **{t['azul']}**")
    return r or ["✅ ¡Color en punto!"]

# ─── AGENTE IA (streaming) ───────────────────────────────────────────────────
def obtener_api_key():
    return os.environ.get("ANTHROPIC_API_KEY") or st.session_state.get("api_key", "")

def stream_maestro(b64_c, b64_m, lab_c, lab_m, sistema, dE, pct,
                   n_c, n_m, std_c, std_m, radio, cal_c, cal_m):
    dL = lab_c[0] - lab_m[0]
    da = lab_c[1] - lab_m[1]
    db = lab_c[2] - lab_m[2]
    tintes = "; ".join(f"{k}: {v}" for k, v in SISTEMAS[sistema].items())

    calidad_c = "buena" if (std_c < 15) else ("aceptable" if std_c < 30 else "alta variación (posible reflejo)")
    calidad_m = "buena" if (std_m < 15) else ("aceptable" if std_m < 30 else "alta variación (posible reflejo)")
    cal_c_txt = "sí, corregido por el usuario" if cal_c else "no"
    cal_m_txt = "sí, corregido por el usuario" if cal_m else "no"

    prompt = f"""Eres un maestro colorimetrista con 30 años de experiencia en pintura de automóviles.

Se te presentan DOS imágenes:
  • Imagen 1 — COLOR OBJETIVO: zona del coche que hay que igualar
  • Imagen 2 — COLOR ACTUAL: la prueba de pintura mezclada en el taller

═══ DATOS COLORIMÉTRICOS (CIE L*a*b*) ═══
  Objetivo (coche)  →  L* = {lab_c[0]:.2f}   a* = {lab_c[1]:+.2f}   b* = {lab_c[2]:+.2f}
  Actual (prueba)   →  L* = {lab_m[0]:.2f}   a* = {lab_m[1]:+.2f}   b* = {lab_m[2]:+.2f}
  Diferencia        →  ΔL = {dL:+.2f}   Δa* = {da:+.2f}   Δb* = {db:+.2f}
  ΔE (CIE76): {dE:.2f}   |   Similitud estimada: {pct:.1f}%

Calidad de muestras: coche={calidad_c} ({n_c} punto/s), prueba={calidad_m} ({n_m} punto/s)
Balance de blancos calibrado manualmente por el usuario: coche={cal_c_txt}, prueba={cal_m_txt}
Radio de muestreo usado: {radio}px por punto (sobre la foto en alta resolución)
Sistema de tintes: {sistema}
Tintes disponibles: {tintes}

REFERENCIA DE SIGNOS (crítico — no confundir):
  ΔL > 0 → prueba MÁS OSCURA que el objetivo → hay que ACLARAR la prueba (añadir blanco/aluminio)
  ΔL < 0 → prueba MÁS CLARA que el objetivo  → hay que OSCURECER la prueba (añadir negro)
  Δa* > 0 → falta ROJO en la prueba           → añadir rojo
  Δa* < 0 → sobra ROJO en la prueba           → añadir verde
  Δb* > 0 → falta AMARILLO en la prueba       → añadir ocre/amarillo
  Δb* < 0 → sobra AMARILLO en la prueba       → añadir azul

Responde con estas secciones:

## 👁️ Diagnóstico Visual
Diferencias percibidas entre las dos imágenes: tono, luminosidad, saturación, si hay efecto metálico o perlado.

## 📊 Interpretación Colorimétrica
Qué significa cada desviación ΔL, Δa*, Δb* para ESTE color concreto. Si la calidad de muestra es mala, avisa.

## 🧪 Receta de Ajuste
Tintes a modificar con cantidades concretas en gotas por 100 ml (o % en peso). Ordena de mayor a menor impacto. Si ΔE > 10, recomienda reformular desde cero.

## ⚠️ Advertencias
Metamerismo, efecto flip/flop en metálicos, influencia del fondo de chapa, calidad de las fotos si es relevante.

## 💡 Truco de Maestro
Un consejo práctico y específico para este caso concreto.

Sé conciso y directo. Habla como un experto a su compañero pintor."""

    client = anthropic.Anthropic(api_key=obtener_api_key())
    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=1800,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64_c}},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64_m}},
                {"type": "text", "text": prompt},
            ],
        }],
    ) as stream:
        yield from stream.text_stream

# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ
# ══════════════════════════════════════════════════════════════════════════════

# ── Barra lateral ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")

    clave_input = st.text_input(
        "🔑 Anthropic API Key", type="password",
        value=st.session_state.get("api_key", ""),
        help="Obtén tu clave en console.anthropic.com",
    )
    if clave_input:
        st.session_state["api_key"] = clave_input

    if obtener_api_key():
        st.success("✓ API Key activa")
    else:
        st.warning("Sin API Key → análisis IA desactivado")

    sistema = st.selectbox("🎨 Sistema de tintes", list(SISTEMAS.keys()))

    radio_muestra = st.slider(
        "📏 Radio de muestra (px)", min_value=5, max_value=80, value=25,
        help="Área de píxeles analizada alrededor del clic, medida sobre la foto en alta resolución. Mayor radio = más promedio, mejor para colores uniformes.",
    )

    st.divider()

    with st.expander("📸 Cómo conseguir mejores fotos"):
        st.markdown("""
**Para mayor precisión:**

1. **Misma iluminación en ambas fotos** — luz natural difusa (día nublado o nave con luz uniforme). Sin flash.
2. **Sin reflejo especular** — mueve la cámara hasta que no veas brillo en la zona.
3. **Perpendicular a la superficie** — fotografía recto, no en ángulo.
4. **Zona plana del coche** — evita curvas y esquinas donde cambia el ángulo de luz.
5. **Prueba sobre cartón o papel gris/blanco neutro** — no uses fondo de color.
6. **Desactiva HDR y Live Photo** — pueden alterar los colores reales.
7. **Mismo momento del día** — la temperatura de color de la luz solar varía mucho.
8. **Sube la foto desde la cámara nativa del móvil** — al tocar "Subir foto" en un teléfono se abre la cámara del sistema, con mejor control de exposición y enfoque que una cámara web del navegador.
9. **¿Luz distinta entre las dos fotos?** — activa "Calibrar con punto neutro" y toca un papel, pared o trapo neutro presente en la propia foto. No hace falta ninguna carta de color.
        """)

    st.divider()
    st.markdown("""**Cómo usar:**
1. Sube la foto del **coche**
2. Haz clic en el color a igualar
3. Sube tu **prueba de pintura**
4. Haz clic en el color de la prueba
5. Pulsa **Analizar con IA** para la receta
> 💡 Varios clics = más precisión (se promedia)""")

# ── Cabecera ──────────────────────────────────────────────────────────────────
st.title("🧪 Spectry Lab")
st.caption("Colorimetría inteligente para pintores de automóviles")

paso1_ok = bool(st.session_state["rgb_coche"])
paso2_ok = bool(st.session_state["rgb_muestra"])
paso3_ok = paso1_ok and paso2_ok

st.markdown(f"""
<div class="steps-row">
  <div class="step-pill {'done' if paso1_ok else 'active'}"><span class="step-num">1</span> Coche</div>
  <div class="step-pill {'done' if paso2_ok else ('active' if paso1_ok else '')}"><span class="step-num">2</span> Prueba</div>
  <div class="step-pill {'active' if paso3_ok else ''}"><span class="step-num">3</span> Resultado</div>
</div>
""", unsafe_allow_html=True)

# ── Bloque de imagen ──────────────────────────────────────────────────────────
def bloque_imagen(titulo, sfx):
    key_pts     = f"puntos_{sfx}"
    key_rgb     = f"rgb_{sfx}"
    key_std     = f"std_{sfx}"
    key_prev    = f"prev_{sfx}"
    key_hash    = f"hash_{sfx}"
    key_imgfull = f"imgfull_{sfx}"
    key_b64     = f"b64_{sfx}"
    key_factor  = f"factor_{sfx}"
    key_calidad = f"calidad_{sfx}"
    key_ilu     = f"ilu_{sfx}"

    with st.container(border=True):
        st.markdown(f"##### {titulo}")
        f = st.file_uploader(
            "Foto", key=f"up_{sfx}",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if not f:
            st.markdown(
                '<p style="color:#B0B7C3;text-align:center;padding:50px 0">📷 Sube una foto</p>',
                unsafe_allow_html=True,
            )
            return

        nuevo_hash = hash_archivo(f)
        if nuevo_hash != st.session_state.get(key_hash):
            img_full = redimensionar(cargar_imagen(f), 2000)
            st.session_state[key_imgfull] = img_full
            st.session_state[key_hash]    = nuevo_hash
            st.session_state[key_pts]     = []
            st.session_state[key_rgb]     = []
            st.session_state[key_std]     = []
            st.session_state[key_prev]    = None
            st.session_state[key_factor]  = None
            st.session_state[key_calidad] = analizar_calidad_captura(img_full)
            st.session_state[key_ilu]     = estimar_iluminante(img_full)
            st.session_state[key_b64]     = a_b64(redimensionar(img_full, 900))
            st.session_state["resultado_ia"] = None

        img_full = st.session_state[key_imgfull]

        for nivel, msg in st.session_state[key_calidad]["avisos"]:
            if nivel == "bad":
                st.warning(msg, icon="⚠️")
            else:
                st.info(msg, icon="💡")

        img_click = redimensionar(img_full, 440)
        escala = img_full.width / img_click.width

        modo_cal = st.checkbox(
            "🎯 Calibrar con punto neutro (gris/blanco presente en la foto)",
            key=f"modocal_{sfx}",
            help="Activa esto y toca un objeto gris o blanco neutro de la propia escena (papel, pared, trapo) para corregir el tono de la luz. No hace falta ninguna carta de color.",
        )

        vis = dibujar_miras(img_click, st.session_state[key_pts])
        st.markdown(
            f'<p class="hint">👇 {"Toca el punto neutro de referencia" if modo_cal else "Haz clic en el color (varios clics = más precisión)"}</p>',
            unsafe_allow_html=True,
        )
        _, col_c, _ = st.columns([1, 10, 1])
        with col_c:
            coord = streamlit_image_coordinates(vis, key=f"cl_{sfx}")

        if coord and coord != st.session_state[key_prev]:
            st.session_state[key_prev] = coord
            x_full = min(img_full.width - 1, int(round(coord["x"] * escala)))
            y_full = min(img_full.height - 1, int(round(coord["y"] * escala)))

            if modo_cal:
                rgb_neutro, _ = muestrear_color(img_full, x_full, y_full, radio_muestra)
                st.session_state[key_factor] = calcular_factor_calibracion(rgb_neutro)
                st.session_state["resultado_ia"] = None
                st.rerun()
            else:
                st.session_state[key_pts].append((coord["x"], coord["y"]))
                rgb, std = muestrear_color(img_full, x_full, y_full, radio_muestra)
                st.session_state[key_rgb].append(rgb)
                st.session_state[key_std].append(std)
                st.session_state["resultado_ia"] = None
                st.rerun()

        if st.session_state[key_factor] is not None:
            fb1, fb2 = st.columns([3, 1])
            with fb1:
                st.markdown('<span class="badge badge-ok">✓ Calibrado con punto neutro</span>', unsafe_allow_html=True)
            with fb2:
                if st.button("Quitar", key=f"quitarcal_{sfx}"):
                    st.session_state[key_factor] = None
                    st.session_state["resultado_ia"] = None
                    st.rerun()

        if not st.session_state[key_rgb]:
            return

        rgb_calibrado = [aplicar_calibracion(c, st.session_state[key_factor]) for c in st.session_state[key_rgb]]

        chips = "".join(
            f'<span class="chip" style="background:{hex_color(c)}" title="Muestra {i+1}"></span>'
            for i, c in enumerate(rgb_calibrado)
        )
        lab = lab_promedio(rgb_calibrado)
        n = len(rgb_calibrado)
        st.markdown(
            f"{chips}&nbsp;<small style='color:#8A94A6'>"
            f"L*={lab[0]:.1f}  a*={lab[1]:+.1f}  b*={lab[2]:+.1f}"
            f" — {n} muestra{'s' if n > 1 else ''}</small>",
            unsafe_allow_html=True,
        )

        std_prom = float(np.mean(st.session_state[key_std]))
        if std_prom < 15:
            st.markdown(f'<span class="badge badge-ok">✓ Muestra uniforme ({std_prom:.0f})</span>', unsafe_allow_html=True)
        elif std_prom < 30:
            st.markdown(f'<span class="badge badge-mid">⚠ Variación moderada ({std_prom:.0f})</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge badge-bad">✗ Alta variación ({std_prom:.0f}) — busca una zona más uniforme</span>', unsafe_allow_html=True)

        lx, ly = st.session_state[key_pts][-1]
        x_full = min(img_full.width - 1, int(round(lx * escala)))
        y_full = min(img_full.height - 1, int(round(ly * escala)))
        with st.expander("🔍 Zoom del último punto"):
            zoom = zoom_punto(img_full, x_full, y_full)
            st.image(zoom, caption=f"Área ampliada (radio {radio_muestra}px sobre la foto original)")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("↩️ Borrar último", key=f"undo_{sfx}",
                         disabled=len(st.session_state[key_pts]) == 0):
                st.session_state[key_pts].pop()
                st.session_state[key_rgb].pop()
                st.session_state[key_std].pop()
                st.session_state[key_prev] = None
                st.session_state["resultado_ia"] = None
                st.rerun()
        with b2:
            if st.button("🗑️ Borrar todos", key=f"del_{sfx}"):
                st.session_state[key_pts]  = []
                st.session_state[key_rgb]  = []
                st.session_state[key_std]  = []
                st.session_state[key_prev] = None
                st.session_state["resultado_ia"] = None
                st.rerun()

# ── Layout 2 columnas ─────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    bloque_imagen("🚗 Color del Coche (objetivo)", "coche")
with col2:
    bloque_imagen("🎨 Prueba del Taller (actual)", "muestra")

# ── Resultados ────────────────────────────────────────────────────────────────
rgb_c_cal = lista_calibrada("coche")
rgb_m_cal = lista_calibrada("muestra")
lab_c = lab_promedio(rgb_c_cal)
lab_m = lab_promedio(rgb_m_cal)

if lab_c and lab_m:
    dL = lab_c[0] - lab_m[0]
    da = lab_c[1] - lab_m[1]
    db = lab_c[2] - lab_m[2]
    dE  = float(np.sqrt(dL**2 + da**2 + db**2))
    pct = max(0.0, 100.0 - dE * 2)

    desc_dE, clase_dE = interpretar_dE(dE)
    color_gauge = {"ok": "#16A34A", "mid": "#D97706", "bad": "#DC2626"}[clase_dE]
    deg = max(0.0, min(100.0, pct)) * 3.6

    st.divider()

    ilu_c = st.session_state.get("ilu_coche")
    ilu_m = st.session_state.get("ilu_muestra")
    if ilu_c is not None and ilu_m is not None:
        dist_ilu = diferencia_iluminantes(ilu_c, ilu_m)
        if dist_ilu > 0.06:
            st.warning(
                "Las dos fotos parecen tomadas con una luz de tono distinto (una más cálida o fría que la otra). "
                "Si puedes, repítelas con la misma iluminación, o usa la calibración por punto neutro en cada una.",
                icon="💡",
            )
        elif dist_ilu > 0.03:
            st.info("La iluminación entre ambas fotos no es idéntica, pero la diferencia es pequeña.", icon="💡")

    avg_c = rgb_promedio(rgb_c_cal)
    avg_m = rgb_promedio(rgb_m_cal)

    st.markdown(f"""
<div class="card" style="text-align:center;">
  <div class="gauge-wrap">
    <div class="gauge" style="--gauge-color:{color_gauge}; --gauge-deg:{deg:.1f};">
      <div class="gauge-inner">
        <div class="gauge-pct">{pct:.0f}%</div>
        <div class="gauge-label">SIMILITUD</div>
      </div>
    </div>
  </div>
  <div class="dE-{clase_dE}" style="font-weight:700;margin-top:12px">{desc_dE}</div>
  <div style="color:#8A94A6;font-size:.78rem;margin-top:6px">
    ΔE = {dE:.2f} &nbsp;·&nbsp; ΔL = {dL:+.2f} &nbsp;·&nbsp; Δa* = {da:+.2f} &nbsp;·&nbsp; Δb* = {db:+.2f}
  </div>
  <div class="vs-row" style="margin-top:20px">
    <div class="vs-swatch-box">
      <div class="vs-swatch" style="background:{hex_color(avg_c)}"></div>
      <div class="vs-label">Objetivo</div>
      <div class="vs-hex">{hex_color(avg_c)}</div>
    </div>
    <div class="vs-versus">VS</div>
    <div class="vs-swatch-box">
      <div class="vs-swatch" style="background:{hex_color(avg_m)}"></div>
      <div class="vs-label">Prueba</div>
      <div class="vs-hex">{hex_color(avg_m)}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Ajuste rápido
    ajustes = ajuste_rapido(sistema, dL, da, db)
    if ajustes[0].startswith("✅"):
        st.success(ajustes[0])
    else:
        with st.expander("🛠️ Ajuste rápido (sin IA)", expanded=True):
            for a in ajustes:
                st.write(f"• {a}")

    # ── Análisis IA ───────────────────────────────────────────────────────────
    st.markdown("### 🤖 Análisis del Maestro Colorimetrista")

    tiene_fotos = bool(st.session_state["b64_coche"] and st.session_state["b64_muestra"])
    tiene_key   = bool(obtener_api_key())

    if not tiene_fotos:
        st.info("Se necesitan las fotos de ambos lados para el análisis con IA.")
    elif not tiene_key:
        st.warning("Configura la API Key de Anthropic en la barra lateral para activar el análisis.")
    else:
        if st.button("🔬 Obtener Receta del Maestro Colorimetrista",
                     type="primary", use_container_width=True):
            st.session_state["resultado_ia"] = None
            std_c_prom = float(np.mean(st.session_state["std_coche"])) if st.session_state["std_coche"] else 0.0
            std_m_prom = float(np.mean(st.session_state["std_muestra"])) if st.session_state["std_muestra"] else 0.0
            try:
                full = st.write_stream(stream_maestro(
                    st.session_state["b64_coche"],
                    st.session_state["b64_muestra"],
                    lab_c, lab_m, sistema, dE, pct,
                    len(st.session_state["rgb_coche"]),
                    len(st.session_state["rgb_muestra"]),
                    std_c_prom, std_m_prom, radio_muestra,
                    st.session_state["factor_coche"] is not None,
                    st.session_state["factor_muestra"] is not None,
                ))
                st.session_state["resultado_ia"] = full
            except anthropic.AuthenticationError:
                st.error("❌ API Key inválida. Comprueba la configuración en la barra lateral.")
            except Exception as e:
                st.error(f"❌ Error al consultar la IA: {e}")
        elif st.session_state["resultado_ia"]:
            st.markdown(st.session_state["resultado_ia"])

elif st.session_state["rgb_coche"] or st.session_state["rgb_muestra"]:
    st.info("👆 Selecciona el color en la otra foto para calcular la diferencia.")
