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
  html, body, [class*="css"] {
    font-size: 14px; background-color: #0E1117; color: #E6E6E6;
  }
  .block-container { padding-top: 1rem; padding-bottom: 3rem; }

  .resultado-box {
    background: #111; border: 2px solid #333; border-radius: 12px;
    padding: 20px; text-align: center; margin: 10px 0;
  }
  .dE-val { font-size: 3.5rem; font-weight: 900; margin: 4px 0; line-height: 1; }
  .verde   { color: #00FF00; }
  .amarillo{ color: #FFD700; }
  .rojo    { color: #FF4444; }

  .chip {
    display: inline-block; width: 26px; height: 26px;
    border-radius: 4px; border: 1px solid #444;
    margin: 2px; vertical-align: middle;
  }
  .swatch {
    width: 100%; height: 70px; border-radius: 8px;
    border: 1px solid #333;
  }
  .hint { color: #666; font-size: .78rem; text-align: center; margin-bottom: 3px; }
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
for clave, valor in {
    "puntos_coche": [], "puntos_muestra": [],
    "rgb_coche": [], "rgb_muestra": [],
    "prev_coche": None, "prev_muestra": None,
    "b64_coche": None, "b64_muestra": None,
    "resultado_ia": None,
}.items():
    if clave not in st.session_state:
        st.session_state[clave] = valor

# ─── HELPERS DE IMAGEN ───────────────────────────────────────────────────────
def cargar_imagen(f):
    f.seek(0)
    return Image.open(f).convert("RGB")

def redimensionar(img, ancho=420):
    r = ancho / img.size[0]
    return img.resize((ancho, int(img.size[1] * r)), Image.Resampling.LANCZOS)

def dibujar_miras(img, puntos):
    arr = np.array(img.copy())
    for i, (x, y) in enumerate(puntos):
        for color, grosor in [((0, 0, 0), 2), ((0, 255, 0), 1)]:
            cv2.line(arr, (x - 15, y), (x + 15, y), color, grosor)
            cv2.line(arr, (x, y - 15), (x, y + 15), color, grosor)
            cv2.circle(arr, (x, y), 10, color, grosor)
        cv2.putText(arr, str(i + 1), (x + 12, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(arr, str(i + 1), (x + 12, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return Image.fromarray(arr)

def muestrear_color(img, x, y, radio=4):
    arr = np.array(img)
    h, w = arr.shape[:2]
    parche = arr[max(0, y - radio):min(h, y + radio + 1),
                 max(0, x - radio):min(w, x + radio + 1)]
    return tuple(parche.mean(axis=(0, 1)).astype(int))

def a_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()

def hex_color(rgb):
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"

# ─── COLORIMETRÍA ─────────────────────────────────────────────────────────────
def rgb_a_lab(rgb):
    norm = np.array([[[rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]]],
                    dtype=np.float32)
    L, a, b = cv2.cvtColor(norm, cv2.COLOR_RGB2Lab)[0][0]
    return float(L), float(a - 128), float(b - 128)

def lab_promedio(lista_rgb):
    if not lista_rgb:
        return None
    return tuple(np.mean([rgb_a_lab(c) for c in lista_rgb], axis=0))

def rgb_promedio(lista_rgb):
    if not lista_rgb:
        return None
    return tuple(int(x) for x in np.mean(lista_rgb, axis=0))

# ─── AJUSTE RÁPIDO (sin IA) ──────────────────────────────────────────────────
def ajuste_rapido(sistema, dL, da, db):
    t = SISTEMAS[sistema]
    consejos = []
    if dL > 2:
        consejos.append(f"Más oscuro → añadir **{t['negro']}**")
    elif dL < -2:
        consejos.append(f"Más claro → añadir **{t['aluminio']}** o **{t['blanco']}**")
    if da > 1.5:
        consejos.append(f"Más rojo → añadir **{t['rojo']}**")
    elif da < -1.5:
        consejos.append(f"Menos rojo → añadir **{t['verde']}**")
    if db > 1.5:
        consejos.append(f"Más amarillo → añadir **{t['ocre']}**")
    elif db < -1.5:
        consejos.append(f"Más azul → añadir **{t['azul']}**")
    return consejos or ["✅ ¡Color en punto!"]

# ─── AGENTE IA (streaming) ───────────────────────────────────────────────────
def obtener_api_key():
    return os.environ.get("ANTHROPIC_API_KEY") or st.session_state.get("api_key", "")

def stream_maestro_colorimetrista(b64_coche, b64_muestra, lab_c, lab_m, sistema, dE, pct):
    dL = lab_c[0] - lab_m[0]
    da = lab_c[1] - lab_m[1]
    db = lab_c[2] - lab_m[2]
    tintes = "; ".join(f"{k}: {v}" for k, v in SISTEMAS[sistema].items())

    prompt = f"""Eres un maestro colorimetrista con 30 años de experiencia en pintura de automóviles.

Se te presentan DOS imágenes:
  • Imagen 1 — COLOR OBJETIVO: zona del coche que hay que igualar
  • Imagen 2 — COLOR ACTUAL: la prueba de pintura mezclada en el taller

═══ DATOS COLORIMÉTRICOS (espacio CIE L*a*b*) ═══
  Objetivo (coche)  →  L* = {lab_c[0]:.2f}   a* = {lab_c[1]:+.2f}   b* = {lab_c[2]:+.2f}
  Actual (prueba)   →  L* = {lab_m[0]:.2f}   a* = {lab_m[1]:+.2f}   b* = {lab_m[2]:+.2f}
  Diferencia        →  ΔL = {dL:+.2f}   Δa* = {da:+.2f}   Δb* = {db:+.2f}
  ΔE (CIELAB): {dE:.2f}   |   Similitud estimada: {pct:.1f}%

Sistema de tintes activo: {sistema}
Tintes disponibles: {tintes}

REFERENCIA DE SIGNOS (muy importante para no confundir):
  ΔL > 0  →  la prueba es más OSCURA que el objetivo  (hay que ACLARAR la prueba)
  ΔL < 0  →  la prueba es más CLARA que el objetivo   (hay que OSCURECER la prueba)
  Δa* > 0 →  la prueba tiene MENOS rojo               (hay que AÑADIR rojo)
  Δa* < 0 →  la prueba tiene MÁS rojo                 (hay que QUITAR rojo / añadir verde)
  Δb* > 0 →  la prueba tiene MENOS amarillo           (hay que AÑADIR amarillo/ocre)
  Δb* < 0 →  la prueba tiene MÁS amarillo             (hay que AÑADIR azul)

Responde con estas secciones:

## 👁️ Diagnóstico Visual
Describe brevemente las diferencias que observas entre las dos imágenes (tono, brillo, saturación, si hay efecto metálico o perlado).

## 📊 Interpretación Colorimétrica
Explica qué significa cada desviación ΔL, Δa*, Δb* para ESTE color concreto en términos prácticos.

## 🧪 Receta de Ajuste
Lista los tintes a modificar con cantidades específicas (gotas por 100 ml o % en peso). Ordena de mayor a menor impacto. Si ΔE > 10, indica que conviene reformular desde cero.

## ⚠️ Advertencias
Menciona posibles problemas: metamerismo, efecto flip/flop en metálicos, influencia del fondo de la chapa, condiciones de iluminación al fotografiar.

## 💡 Truco de Maestro
Un consejo práctico y específico para este caso concreto basado en tu experiencia.

Sé conciso y habla directamente al pintor como lo haría un compañero experto."""

    client = anthropic.Anthropic(api_key=obtener_api_key())
    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=1800,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64_coche}},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64_muestra}},
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

    st.divider()
    st.markdown("""**Cómo usar:**
1. Sube la foto del **coche**
2. Haz clic en el color a igualar
3. Sube tu **prueba de pintura**
4. Haz clic en el color de la prueba
5. Pulsa **Analizar con IA** para la receta

> 💡 Varios clics = mayor precisión (se promedia el color)""")

# ── Bloque de imagen reutilizable ─────────────────────────────────────────────
def bloque_imagen(titulo, sfx, key_puntos, key_rgb, key_prev, key_b64):
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

    img = redimensionar(cargar_imagen(f))
    st.session_state[key_b64] = a_b64(img)

    vis = dibujar_miras(img, st.session_state[key_puntos])
    st.markdown(
        '<p class="hint">👇 Haz clic en el color (varios clics = mayor precisión)</p>',
        unsafe_allow_html=True,
    )
    _, col_c, _ = st.columns([1, 10, 1])
    with col_c:
        coord = streamlit_image_coordinates(vis, key=f"cl_{sfx}")

    if coord and coord != st.session_state[key_prev]:
        st.session_state[key_prev] = coord
        st.session_state[key_puntos].append((coord["x"], coord["y"]))
        st.session_state[key_rgb].append(muestrear_color(img, coord["x"], coord["y"]))
        st.rerun()

    if st.session_state[key_rgb]:
        c1, c2 = st.columns([6, 1])
        with c1:
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
        with c2:
            if st.button("🗑️", key=f"del_{sfx}", help="Borrar todas las muestras"):
                st.session_state[key_puntos] = []
                st.session_state[key_rgb] = []
                st.session_state[key_prev] = None
                st.rerun()

# ── Layout 2 columnas ─────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    bloque_imagen("🚗 Color del Coche (objetivo)", "c",
                  "puntos_coche", "rgb_coche", "prev_coche", "b64_coche")
with col2:
    bloque_imagen("🎨 Prueba del Taller (actual)", "m",
                  "puntos_muestra", "rgb_muestra", "prev_muestra", "b64_muestra")

# ── Resultados ────────────────────────────────────────────────────────────────
lab_c = lab_promedio(st.session_state["rgb_coche"])
lab_m = lab_promedio(st.session_state["rgb_muestra"])

if lab_c and lab_m:
    dL = lab_c[0] - lab_m[0]
    da = lab_c[1] - lab_m[1]
    db = lab_c[2] - lab_m[2]
    dE = float(np.sqrt(dL**2 + da**2 + db**2))
    pct = max(0.0, 100.0 - dE * 2)

    clase = "rojo"
    if pct > 88:
        clase = "amarillo"
    if pct > 95:
        clase = "verde"

    st.divider()

    # Swatches de comparación visual
    avg_rgb_c = rgb_promedio(st.session_state["rgb_coche"])
    avg_rgb_m = rgb_promedio(st.session_state["rgb_muestra"])
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(
            f'<div class="swatch" style="background:{hex_color(avg_rgb_c)}"></div>'
            f'<p style="color:#666;font-size:.8rem;text-align:center;margin-top:4px">Objetivo (coche)</p>',
            unsafe_allow_html=True,
        )
    with sc2:
        st.markdown(
            f'<div class="swatch" style="background:{hex_color(avg_rgb_m)}"></div>'
            f'<p style="color:#666;font-size:.8rem;text-align:center;margin-top:4px">Actual (prueba)</p>',
            unsafe_allow_html=True,
        )

    # Caja de similitud
    st.markdown(f"""
<div class="resultado-box">
  <div style="color:#555;letter-spacing:2px;font-size:.8rem">SIMILITUD CROMÁTICA</div>
  <div class="dE-val {clase}">{pct:.1f}%</div>
  <div style="color:#777;font-size:.82rem;margin-top:6px">
    ΔE = {dE:.2f} &nbsp;·&nbsp;
    ΔL = {dL:+.2f} &nbsp;·&nbsp;
    Δa* = {da:+.2f} &nbsp;·&nbsp;
    Δb* = {db:+.2f}
  </div>
</div>
""", unsafe_allow_html=True)

    # Ajuste rápido sin IA
    ajustes = ajuste_rapido(sistema, dL, da, db)
    if ajustes[0].startswith("✅"):
        st.success(ajustes[0])
    else:
        with st.expander("🛠️ Ajuste rápido (sin IA)", expanded=True):
            for a in ajustes:
                st.write(f"• {a}")

    # Análisis IA
    st.markdown("### 🤖 Análisis del Maestro Colorimetrista")

    tiene_fotos = bool(st.session_state["b64_coche"] and st.session_state["b64_muestra"])
    tiene_key = bool(obtener_api_key())

    if not tiene_fotos:
        st.info("Se necesitan las fotos de ambos lados para el análisis con IA.")
    elif not tiene_key:
        st.warning("Configura la API Key de Anthropic en la barra lateral para activar el análisis.")
    else:
        if st.button("🔬 Obtener Receta del Maestro Colorimetrista",
                     type="primary", use_container_width=True):
            st.session_state["resultado_ia"] = None
            try:
                texto_completo = st.write_stream(stream_maestro_colorimetrista(
                    st.session_state["b64_coche"],
                    st.session_state["b64_muestra"],
                    lab_c, lab_m, sistema, dE, pct,
                ))
                st.session_state["resultado_ia"] = texto_completo
            except anthropic.AuthenticationError:
                st.error("❌ API Key inválida. Comprueba la configuración en la barra lateral.")
            except Exception as e:
                st.error(f"❌ Error al consultar la IA: {e}")
        elif st.session_state["resultado_ia"]:
            st.markdown(st.session_state["resultado_ia"])

elif st.session_state["rgb_coche"] or st.session_state["rgb_muestra"]:
    st.info("👆 Selecciona el color en la otra foto para calcular la diferencia.")
