import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 1. CONFIGURACIÓN Y ESTILO (V8 - Márgenes y Directo)
# ==========================================
st.set_page_config(page_title="Spectry Lab", layout="wide", page_icon="🧪")

st.markdown("""
    <style>
        /* Forzar scroll */
        .main { overflow-y: scroll !important; }
        
        /* Ajustes de texto */
        html, body, [class*="css"] { font-size: 14px; background-color: #0E1117; color: #E6E6E6; }
        .block-container { padding-top: 1rem; padding-bottom: 3rem; }
        
        /* CAJA DE RESULTADO */
        .resultado-final {
            background-color: #111;
            border: 2px solid #333;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            margin-top: 10px;
        }
        .similitud-valor {
            font-size: 3rem;
            font-weight: 900;
            margin: 0;
            text-shadow: 0px 0px 10px rgba(0,0,0,0.5);
        }
        
        /* Colores del semáforo */
        .verde { color: #00FF00; }
        .amarillo { color: #FFD700; }
        .rojo { color: #FF4444; }

        /* Instrucciones visuales */
        .instruccion {
            color: #888;
            font-size: 0.8rem;
            text-align: center;
            margin-bottom: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATOS
# ==========================================
SISTEMAS = {
    "BESA Urki-Mix": {"negro": "9005", "blanco": "Blanco", "alu": "Aluminio", "ocre": "Ocre", "rojo": "Rojo"},
    "NOVOL Spectral": {"negro": "SB-1000", "blanco": "SB-2000", "alu": "B-810", "ocre": "B-Ocre", "rojo": "B-Rojo"}
}

# ==========================================
# 3. MOTOR GRÁFICO ROBUSTO (FIX INCLUIDO)
# ==========================================

# Inicializar sesión
if 'coord_coche' not in st.session_state: st.session_state.coord_coche = None
if 'coord_muestra' not in st.session_state: st.session_state.coord_muestra = None
if 'color_coche' not in st.session_state: st.session_state.color_coche = None
if 'color_muestra' not in st.session_state: st.session_state.color_muestra = None

def load_image_safe(uploaded_file):
    """Carga segura que rebobina el archivo para evitar errores"""
    uploaded_file.seek(0)
    return Image.open(uploaded_file).convert("RGB")

def resize_image(image, width=400):
    w_percent = (width / float(image.size[0]))
    h_size = int((float(image.size[1]) * float(w_percent)))
    return image.resize((width, h_size), Image.Resampling.LANCZOS)

def dibujar_mira_laser(image, x, y):
    """Dibuja una mira de francotirador en el punto seleccionado"""
    img_np = np.array(image)
    
    # 1. Cruz negra (fondo para contraste)
    cv2.line(img_np, (x-15, y), (x+15, y), (0,0,0), 1)
    cv2.line(img_np, (x, y-15), (x, y+15), (0,0,0), 1)
    
    # 2. Cruz verde neón (frente)
    cv2.line(img_np, (x-15, y), (x+15, y), (0,255,0), 1)
    cv2.line(img_np, (x, y-15), (x, y+15), (0,255,0), 1)
    
    # 3. Círculo
    cv2.circle(img_np, (x, y), 10, (0, 255, 0), 1)
    
    return Image.fromarray(img_np)

def rgb_to_lab(rgb):
    norm = np.array([[[rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0]]], dtype=np.float32)
    lab = cv2.cvtColor(norm, cv2.COLOR_RGB2Lab)
    return lab[0][0][0], lab[0][0][1]-128, lab[0][0][2]-128

def generar_consejos(sistema, dL, da, db):
    tintes = SISTEMAS[sistema]
    consejos = []
    if dL > 2.0: consejos.append(f"🌑 OSCURO: +{tintes['alu']}")
    elif dL < -2.0: consejos.append(f"☀️ CLARO: +{tintes['negro']}")
    if da > 1.5: consejos.append(f"🔥 Falta ROJO: +{tintes['rojo']}")
    elif da < -1.5: consejos.append(f"🌿 Sobra ROJO: +Verde")
    if db > 1.5: consejos.append(f"🌻 Falta AMARILLO: +{tintes['ocre']}")
    elif db < -1.5: consejos.append(f"🌊 Sobra AMARILLO: +Azul")
    if not consejos: return ["✅ ¡PERFECTO!"]
    return consejos

# ==========================================
# 4. INTERFAZ DIRECTA (V8 LAYOUT)
# ==========================================
st.title("Spectry Lab")
marca = st.selectbox("Sistema", list(SISTEMAS.keys()), label_visibility="collapsed")

# Función auxiliar para renderizar cada bloque con márgenes
def bloque_imagen(titulo, key_suffix, session_coord, session_color):
    st.subheader(titulo)
    uploaded = st.file_uploader(f"Subir {titulo}", key=f"up_{key_suffix}", label_visibility="collapsed")
    
    if uploaded:
        # 1. Cargar y Redimensionar (Seguro)
        img_raw = load_image_safe(uploaded)
        img_raw = resize_image(img_raw, width=400)
        
        # 2. Si ya hay un clic guardado, dibujamos la mira ANTES de mostrarla
        img_to_show = img_raw.copy()
        current_coord = st.session_state[session_coord]
        
        if current_coord:
            img_to_show = dibujar_mira_laser(img_to_show, current_coord['x'], current_coord['y'])

        # 3. Mostrar imagen clickable
        st.markdown('<p class="instruccion">👇 Toca el color (Usa los lados para bajar)</p>', unsafe_allow_html=True)
        
        # MÁRGENES DE SEGURIDAD PARA SCROLL
        c_left, c_center, c_right = st.columns([1, 10, 1]) 
        
        with c_center:
            # Componente de clic
            new_val = streamlit_image_coordinates(img_to_show, key=f"click_{key_suffix}")
            
            # 4. Lógica de actualización instantánea
            if new_val and new_val != current_coord:
                st.session_state[session_coord] = new_val
                # Guardamos el color
                px = img_raw.getpixel((new_val['x'], new_val['y']))
                st.session_state[session_color] = rgb_to_lab(px)
                st.rerun() # Recarga inmediata para pintar la mira

# --- LAYOUT PRINCIPAL ---
col1, col2 = st.columns(2)

with col1:
    bloque_imagen("🚗 Coche", "c", "coord_coche", "color_coche")

with col2:
    bloque_imagen("🎨 Muestra", "m", "coord_muestra", "color_muestra")

# ==========================================
# 5. RESULTADOS
# ==========================================
if st.session_state.color_coche and st.session_state.color_muestra:
    Lc, ac, bc = st.session_state.color_coche
    Lm, am, bm = st.session_state.color_muestra
    dL, da, db = Lc-Lm, ac-am, bc-bm
    dE = np.sqrt(dL**2 + da**2 + db**2)
    
    porcentaje = max(0, 100 - (dE * 2))
    color_class = "rojo"
    if porcentaje > 88: color_class = "amarillo"
    if porcentaje > 95: color_class = "verde"

    st.markdown("---")
    
    # CAJA VISUAL DE RESULTADO
    st.markdown(f"""
    <div class="resultado-final">
        <div style="color:#888; letter-spacing: 2px; margin-bottom:5px;">COINCIDENCIA</div>
        <div class="similitud-valor {color_class}">{porcentaje:.1f}%</div>
        <div style="margin-top:10px; font-size:1.2rem; color:#FFF;">
            L: <b>{dL:.1f}</b> &nbsp;|&nbsp; a: <b>{da:.1f}</b> &nbsp;|&nbsp; b: <b>{db:.1f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # RECETA DE AJUSTE
    consejos = generar_consejos(marca, dL, da, db)
    st.markdown(f"### 🛠️ {' | '.join(consejos)}")

elif st.session_state.coord_coche or st.session_state.coord_muestra:
    st.info("👆 Te falta seleccionar el otro color para calcular.")