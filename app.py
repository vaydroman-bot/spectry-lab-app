import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import anthropic
import base64
import io
import os

# ─── CONFIGURACIÓN ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Spectry Lab", layout="wide", page_icon="🧪")

st.markdown("""
<style>
  .main { overflow-y: scroll !important; }
  html, body, [class*="css"] { font-size: 14px; background-color: #0E1117; color: #E6E6E6; }
  .block-container { padding-top: 1rem; padding-bottom: 3rem; }

  .resultado-box {
    background: #111; border: 2px solid #333; border-radius: 12px;
    padding: 20px; text-align: center; margin: 10px 0;
  }
  .dE-val { font-size: 3.5rem; font-weight: 900; margin: 4px 0; line-height: 1; }
  .verde    { color: #00FF00; }
  .amarillo { color: #FFD700; }
  .rojo     { color: #FF4444; }

  .chip {
    display: inline-block; width: 26px; height: 26px;
    border-radius: 4px; border: 1px solid #444; margin: 2px; vertical-align: middle;
  }
  .swatch { width: 100%; height: 70px; border-radius: 8px; border: 1px solid #333; }
  .hint { color: #666; font-size: .78rem; text-align: center; margin-bottom: 3px; }
  .q-ok  { color: #00cc44; font-size: .75rem; }
  .q-mid { color: #FFD700; font-size: .75rem; }
  .q-bad { color: #FF4444; font-size: .75rem; }
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
for k, v in {
    "puntos_coche": [], "puntos_muestra": [],
    "rgb_coche": [],    "rgb_muestra": [],
    "std_coche": [],    "std_muestra": [],
    "prev_coche": None, "prev_muestra": None,
    "b64_coche": None,  "b64_muestra": None,
    "file_coche": None, "file_muestra": None,
    "resultado_ia": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── HELPERS DE IMAGEN ───────────────────────────────────────────────────────
def cargar_imagen(f):
    f.seek(0)
    return Image.open(f).convert("RGB")

def redimensionar(img, ancho=420):
    # FIX: no forzar upscale si la imagen es más pequeña
    if img.size[0] <= ancho:
        return img.copy()
    r = ancho / img.size[0]
    return img.resize((ancho, int(img.size[1] * r)), Image.Resampling.LANCZOS)

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
    pixeles = parche.reshape(-1, 3)
    # FIX: mediana (más robusta a reflejos que la media) + redondeo correcto
    rgb = tuple(np.median(pixeles, axis=0).round().astype(int))
    std = float(pixeles.std())
    return rgb, std

def zoom_punto(img, x, y, radio_zoom=50, tam=180):
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
        return "Diferencia imperceptible al ojo humano", "verde"
    elif dE < 2.0:
        return "Solo detectable por expertos entrenados", "verde"
    elif dE < 3.5:
        return "Perceptible — umbral de aceptación industrial", "amarillo"
    elif dE < 5.0:
        return "Diferencia clara — ajuste necesario", "amarillo"
    elif dE < 10.0:
        return "Diferencia notable — ajuste importante", "rojo"
    else:
        return "Colores muy distintos — valorar reformular", "rojo"

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
    # FIX CRÍTICO: los signos estaban invertidos en la versión anterior
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
                   n_c, n_m, std_c, std_m, radio):
    dL = lab_c[0] - lab_m[0]
    da = lab_c[1] - lab_m[1]
    db = lab_c[2] - lab_m[2]
    tintes = "; ".join(f"{k}: {v}" for k, v in SISTEMAS[sistema].items())

    cal_c = "buena" if (std_c < 15) else ("aceptable" if std_c < 30 else "alta variación (posible reflejo)")
    cal_m = "buena" if (std_m < 15) else ("aceptable" if std_m < 30 else "alta variación (posible reflejo)")

    prompt = f"""Eres un maestro colorimetrista con 30 años de experiencia en pintura de automóviles.

Se te presentan DOS imágenes:
  • Imagen 1 — COLOR OBJETIVO: zona del coche que hay que igualar
  • Imagen 2 — COLOR ACTUAL: la prueba de pintura mezclada en el taller

═══ DATOS COLORIMÉTRICOS (CIE L*a*b*) ═══
  Objetivo (coche)  →  L* = {lab_c[0]:.2f}   a* = {lab_c[1]:+.2f}   b* = {lab_c[2]:+.2f}
  Actual (prueba)   →  L* = {lab_m[0]:.2f}   a* = {lab_m[1]:+.2f}   b* = {lab_m[2]:+.2f}
  Diferencia        →  ΔL = {dL:+.2f}   Δa* = {da:+.2f}   Δb* = {db:+.2f}
  ΔE (CIE76): {dE:.2f}   |   Similitud estimada: {pct:.1f}%

Calidad de muestras: coche={cal_c} ({n_c} punto/s), prueba={cal_m} ({n_m} punto/s)
Radio de muestreo usado: {radio}px por punto
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
st.title("🧪 Spectry Lab")
st.caption("Colorimetría inteligente para pintores de automóviles")

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
        "📏 Radio de muestra (px)", min_value=2, max_value=25, value=6,
        help="Área de píxeles analizada alrededor del clic. Mayor radio = más promedio, mejor para colores uniformes.",
    )

    st.divider()

    with st.expander("📸 Cómo conseguir mejores fotos"):
        st.markdown("""
**Para mayor precisión:**

1. **Misma iluminación en ambas fotos** — luz natural difusa (día nublado o nave con luz uniforme). Sin flash.
2. **Sin reflejo especular** — mueve la cámara hasta que no veas brillo en la zona.
3. **Perpendicular a la superficie** — fotografía recto, no en ángulo.
4. **Zona plana del coche** — evita curvas y esquinas donde cambia el ángulo de luz.
5. **Prueba sobre cartón gris neutro** — no uses fondo blanco ni negro.
6. **Desactiva HDR y Live Photo** — pueden alterar los colores reales.
7. **Mismo momento del día** — la temperatura de color de la luz solar varía mucho.
        """)

    st.divider()
    st.markdown("""**Cómo usar:**
1. Sube la foto del **coche**
2. Haz clic en el color a igualar
3. Sube tu **prueba de pintura**
4. Haz clic en el color de la prueba
5. Pulsa **Analizar con IA** para la receta
> 💡 Varios clics = más precisión (se promedia)""")

# ── Bloque de imagen ──────────────────────────────────────────────────────────
def bloque_imagen(titulo, sfx, key_pts, key_rgb, key_std, key_prev, key_b64, key_file):
    st.subheader(titulo)
    f = st.file_uploader(
        "Foto", key=f"up_{sfx}",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if not f:
        st.markdown(
            '<p style="color:#444;text-align:center;padding:50px 0">📷 Sube una foto</p>',
            unsafe_allow_html=True,
        )
        return

    # FIX: detectar cambio de imagen y limpiar muestras antiguas
    if f.name != st.session_state.get(key_file):
        st.session_state[key_pts]  = []
        st.session_state[key_rgb]  = []
        st.session_state[key_std]  = []
        st.session_state[key_prev] = None
        st.session_state[key_file] = f.name
        st.session_state["resultado_ia"] = None  # invalidar análisis anterior

    img = redimensionar(cargar_imagen(f))
    st.session_state[key_b64] = a_b64(img)

    vis = dibujar_miras(img, st.session_state[key_pts])
    st.markdown(
        '<p class="hint">👇 Haz clic en el color (múltiples clics = mayor precisión)</p>',
        unsafe_allow_html=True,
    )
    _, col_c, _ = st.columns([1, 10, 1])
    with col_c:
        coord = streamlit_image_coordinates(vis, key=f"cl_{sfx}")

    if coord and coord != st.session_state[key_prev]:
        st.session_state[key_prev] = coord
        st.session_state[key_pts].append((coord["x"], coord["y"]))
        rgb, std = muestrear_color(img, coord["x"], coord["y"], radio_muestra)
        st.session_state[key_rgb].append(rgb)
        st.session_state[key_std].append(std)
        # FIX: invalidar análisis al añadir nueva muestra
        st.session_state["resultado_ia"] = None
        st.rerun()

    if not st.session_state[key_rgb]:
        return

    # Chips de color + valores LAB medios
    chips = "".join(
        f'<span class="chip" style="background:{hex_color(c)}" title="Muestra {i+1}"></span>'
        for i, c in enumerate(st.session_state[key_rgb])
    )
    lab = lab_promedio(st.session_state[key_rgb])
    n = len(st.session_state[key_rgb])
    st.markdown(
        f"{chips}&nbsp;<small style='color:#666'>"
        f"L*={lab[0]:.1f}  a*={lab[1]:+.1f}  b*={lab[2]:+.1f}"
        f" — {n} muestra{'s' if n > 1 else ''}</small>",
        unsafe_allow_html=True,
    )

    # Indicador de calidad de la muestra
    std_prom = float(np.mean(st.session_state[key_std]))
    if std_prom < 15:
        st.markdown(f'<p class="q-ok">✓ Muestra uniforme (variación {std_prom:.0f})</p>',
                    unsafe_allow_html=True)
    elif std_prom < 30:
        st.markdown(f'<p class="q-mid">⚠ Variación moderada ({std_prom:.0f}) — puede haber reflejo o textura</p>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="q-bad">✗ Alta variación ({std_prom:.0f}) — elige una zona más uniforme y sin brillo</p>',
                    unsafe_allow_html=True)

    # Zoom del último punto seleccionado
    lx, ly = st.session_state[key_pts][-1]
    with st.expander("🔍 Zoom del último punto"):
        zoom = zoom_punto(img, lx, ly)
        st.image(zoom, caption=f"Área ampliada (radio {radio_muestra}px)")

    # Botones de gestión
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
    bloque_imagen("🚗 Color del Coche (objetivo)", "c",
                  "puntos_coche", "rgb_coche", "std_coche",
                  "prev_coche", "b64_coche", "file_coche")
with col2:
    bloque_imagen("🎨 Prueba del Taller (actual)", "m",
                  "puntos_muestra", "rgb_muestra", "std_muestra",
                  "prev_muestra", "b64_muestra", "file_muestra")

# ── Resultados ────────────────────────────────────────────────────────────────
lab_c = lab_promedio(st.session_state["rgb_coche"])
lab_m = lab_promedio(st.session_state["rgb_muestra"])

if lab_c and lab_m:
    dL = lab_c[0] - lab_m[0]
    da = lab_c[1] - lab_m[1]
    db = lab_c[2] - lab_m[2]
    dE  = float(np.sqrt(dL**2 + da**2 + db**2))
    pct = max(0.0, 100.0 - dE * 2)

    desc_dE, clase_dE = interpretar_dE(dE)

    st.divider()

    # Swatches de comparación
    avg_c = rgb_promedio(st.session_state["rgb_coche"])
    avg_m = rgb_promedio(st.session_state["rgb_muestra"])
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(
            f'<div class="swatch" style="background:{hex_color(avg_c)}"></div>'
            f'<p style="color:#666;font-size:.8rem;text-align:center;margin-top:4px">'
            f'Objetivo — {hex_color(avg_c)}</p>',
            unsafe_allow_html=True,
        )
    with sc2:
        st.markdown(
            f'<div class="swatch" style="background:{hex_color(avg_m)}"></div>'
            f'<p style="color:#666;font-size:.8rem;text-align:center;margin-top:4px">'
            f'Actual — {hex_color(avg_m)}</p>',
            unsafe_allow_html=True,
        )

    # Caja de similitud
    st.markdown(f"""
<div class="resultado-box">
  <div style="color:#555;letter-spacing:2px;font-size:.8rem">SIMILITUD CROMÁTICA</div>
  <div class="dE-val {clase_dE}">{pct:.1f}%</div>
  <div style="color:#999;font-size:.85rem;margin-top:4px">{desc_dE}</div>
  <div style="color:#555;font-size:.78rem;margin-top:8px">
    ΔE = {dE:.2f} &nbsp;·&nbsp;
    ΔL = {dL:+.2f} &nbsp;·&nbsp;
    Δa* = {da:+.2f} &nbsp;·&nbsp;
    Δb* = {db:+.2f}
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
