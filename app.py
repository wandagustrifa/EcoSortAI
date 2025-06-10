import streamlit as st
import time
import os
import gdown
import requests
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # Nonaktif GPU
    from tensorflow.keras.models import load_model
except ImportError:
    st.error("TensorFlow tidak terinstal dengan benar. Silakan instal ulang dengan perintah: pip install tensorflow-cpu")
    st.stop()
except Exception as e:
    st.error(f"Error TensorFlow: {str(e)}")
    st.stop()

from PIL import Image
import numpy as np
import io

# --- Configuration Streamlit ---
st.set_page_config(
    page_title="EcoSort - AI Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Reset dan Base Styling */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Container */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Header Section */
    .hero-section {
        text-align: center;
        padding: 3rem 1rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .hero-title {
        font-size: clamp(2.5rem, 5vw, 4rem);
        font-weight: 700;
        background: linear-gradient(135deg, #fff 0%, #f0f8ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Project Description */
    .project-description {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        color: #333;
        line-height: 1.7;
        text-align: center;
    }
    .project-description h2 {
        font-size: 1.8rem;
        color: #4f46e5;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .project-description p {
        font-size: 1.05rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Card Styling (for the main glass-card wrapping sections) */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    }
    .upload-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #e6f2ff 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 2px dashed #4f46e5;
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between; /* Mendorong konten ke atas dan tombol ke bawah */
        align-items: center;
        height: 100%;
        width: 100%;
    }
    .button-container {
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: auto;
        height: 60px; /* Tinggi yang konsisten untuk area tombol */
    }
    /* Upload Section - Grid Layout */
    .upload-section {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin-bottom: 2rem;
    }
    
    /* Styling for st.container used as upload cards */
    /* This targets the stContainer div that is a direct child of the inner stVerticalBlock (column) */
    .upload-section > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {
        background: linear-gradient(135deg, #f8f9ff 0%, #e6f2ff 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 2px dashed #4f46e5;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        display: flex; /* Make it a flex container */
        flex-direction: column; /* Stack children vertically */
        justify-content: space-between; /* Push content and uploader to ends */
        align-items: center; /* Center horizontally */
        height: 100%; /* Ensure both cards are same height in grid */
    }

    .upload-section > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: left 0.5s;
    }
    
    .upload-section > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"]:hover::before {
        left: 100%;
    }
    
    .upload-section > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"]:hover {
        border-color: #6366f1;
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.15);
    }

    .upload-card-content {
        margin-bottom: 1rem; /* Space between content and uploader */
    }
    
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .upload-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .upload-subtitle {
        color: #64748b;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 1rem;
        background: rgba(255, 255, 255, 0.7);
        padding: 0.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Custom File Uploader */
    .stFileUploader {
        width: 100%;
        margin-top: auto; /* Push to bottom of flex container */
    }
    /* Hide the default Streamlit file uploader button and text */
    .stFileUploader > div:first-child > div:first-child > button, 
    .stFileUploader > div:first-child > div:first-child > div {
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }
    .stFileUploader label, .stButton > button {
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        width: 180px !important; /* Lebar tetap yang sama */
        height: 45px !important; /* Tinggi tetap yang sama */
        margin: 0 !important;
        box-sizing: border-box !important;
    }
    .stFileUploader label {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
    }

    /* Custom Camera Input */
    .stCameraInput {
        width: 100%;
        margin-top: auto; /* Push to bottom of flex container */
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    /* Style for the "Activate Camera" button generated by st.button (for initial activation) */
    .stButton > button { 
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3) !important;
    }

    /* Style for the camera video feed and captured image from st.camera_input */
    .stCameraInput video, .stCameraInput img { 
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem; /* Space below video/image */
        max-width: 100%;
        height: auto;
    }
    
    /* Results Section */
    .results-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        margin-top: 2rem;
    }
    
    @media (max-width: 768px) {
        .results-container {
            grid-template-columns: 1fr;
        }
        .upload-section {
            grid-template-columns: 1fr;
        }
    }
    
    /* Styling for st.container used as image preview in results */
    /* This targets the stContainer div that is a direct child of the inner stVerticalBlock (column) */
    .results-container > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        aspect-ratio: 1; /* Maintain aspect ratio for the box */
        background: #f8fafc;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1rem;
        height: auto; 
        width: 100%; 
    }

    /* Ensure image within the preview container is styled correctly */
    .results-container > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        border-radius: 12px;
    }
    /* Ensure the image and caption are centered within the new styled container. */
    .results-container > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] .stImage > div { /* This is the div containing both img and caption */
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 100%; 
        height: 100%; 
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 2rem;
        border-left: 6px solid; /* Dynamic color */
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
    }
    
    .prediction-result {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
    }
    
    .confidence-bar {
        background: #e2e8f0;
        border-radius: 10px;
        height: 12px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-in-out;
    }
    
    .tips-section {
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        border-left: 4px solid; /* Dynamic color set in HTML */
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }

    .tips-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .tips-content {
        line-height: 1.6;
        font-size: 1rem;
    }
    /* Loading Animation */
    .loading-container { /* This class is now used only for the custom styling of the spinner container */
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 4px solid #e2e8f0;
        border-top: 4px solid #4f46e5;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 1rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Alert Styling */
    .custom-alert {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid;
        font-weight: 500;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        color: #15803d;
        border-left-color: #22c55e;
    }
    
    .alert-info {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        color: #1e40af;
        border-left-color: #3b82f6;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%);
        color: #92400e;
        border-left-color: #f59e0b;
    }
    
    /* Hide Streamlit Elements */
    .stDeployButton {display: none;}
    .stDecoration {display: none;}
    #MainMenu {visibility: hidden;}
    .stHeader {display: none;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Model ---
@st.cache_resource
def load_ml_model():
    model_path = 'model_sampah_vgg16.keras' 
    gdrive_file_id = "1lWx7TBcjxxFO3MOUWKEW7oUPVepWxgqN" 

    try:
        if os.path.exists(model_path) and os.path.getsize(model_path) < 100000000:
            st.warning(f"File '{model_path}' ditemukan tetapi ukurannya terlalu kecil ({os.path.getsize(model_path)} bytes). Mengunduh ulang...")
            os.remove(model_path) 

        if not os.path.exists(model_path):
            st.info(f"Mengunduh model dari Google Drive (ID: {gdrive_file_id}) ke '{model_path}'...")
            gdown.download(id=gdrive_file_id, output=model_path, quiet=False, fuzzy=True)

        # Muat model
        model = load_model(model_path)
        return model

    except Exception as e:
        st.error(f"**GAGAL MEMUAT ATAU MENGUNDUH MODEL!** Detail: {e}")
        st.error("Pastikan ID file Google Drive benar, file publik, dan ada cukup memori/disk di lingkungan deployment.")
        st.stop()

# Inisialisasi session state
if 'show_camera' not in st.session_state:
    st.session_state.show_camera = False
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'camera_file_buffer' not in st.session_state:
    st.session_state.camera_file_buffer = None


# Load model
model = load_ml_model()

# --- Define Class Label ---
class_labels = {
    0: 'Anorganik Daur Ulang',
    1: 'Anorganik Tidak Daur Ulang',
    2: 'B3 (Bahan Berbahaya dan Beracun)',
    3: 'Organik'
}

# Color for each category
category_colors = {
    'Anorganik Daur Ulang': '#22c55e', # Green
    'Anorganik Tidak Daur Ulang': '#ef4444', # Red
    'B3 (Bahan Berbahaya dan Beracun)': '#f59e0b', # Orange
    'Organik': '#8b5cf6' # Purple
}

# --- Function Prediction ---
def predict_image(image_file, model, class_labels):
    try:
        img = Image.open(image_file).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class_idx] * 100
        predicted_label = class_labels.get(predicted_class_idx, "Tidak Diketahui")
        
        return predicted_label, confidence, img
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
        return None, None, None

def get_tips(category, base_color):
    # Function to convert from hexadesimal to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # Function to convert RGB 
    def rgb_to_hex(rgb_color):
        return '#%02x%02x%02x' % rgb_color

    # Function to brightest color
    def lighten_color(hex_color, amount):
        r, g, b = hex_to_rgb(hex_color)
        r = min(255, int(r + (255 - r) * amount))
        g = min(255, int(g + (255 - g) * amount)) 
        b = min(255, int(b + (255 - b) * amount))
        return rgb_to_hex((r, g, b))

    # Calculate Background color
    bg_color_start = lighten_color(base_color, 0.7) 
    bg_color_end = lighten_color(base_color, 0.9) 

    tips_data = {
        'Anorganik Daur Ulang': {
            'icon': '♻️',
            'title': 'Tips Daur Ulang Anorganik',
            'content': 'Sampah ini dapat didaur ulang! Pastikan membersihkannya dan memilahnya dengan benar. Termasuk botol plastik bersih, kertas, kardus, dan kaleng. Dengan daur ulang, kita mengurangi kebutuhan bahan baku baru!'
        },
        'Anorganik Tidak Daur Ulang': {
            'icon': '🗑️',
            'title': 'Tips Sampah Anorganik Umum',
            'content': 'Sampah ini umumnya sulit atau tidak dapat didaur ulang. Buanglah ke tempat sampah umum. Upayakan mengurangi penggunaan produk yang menghasilkan sampah jenis ini. Contoh: styrofoam, plastik kemasan berlapis.'
        },
        'B3 (Bahan Berbahaya dan Beracun)': {
            'icon': '⚠️',
            'title': 'PERHATIAN! Limbah B3',
            'content': 'JANGAN dibuang ke tempat sampah biasa! Buanglah ke fasilitas khusus penampungan limbah B3 untuk menghindari pencemaran lingkungan dan bahaya kesehatan. Contoh: baterai bekas, lampu TL, obat kadaluarsa.'
        },
        'Organik': {
            'icon': '🌱',
            'title': 'Tips Pengelolaan Sampah Organik',
            'content': 'Sampah organik dapat diolah menjadi kompos atau pupuk. Cara fantastis untuk mengurangi limbah dan menyuburkan tanah! Pertimbangkan membuat kompos di rumah. Contoh: sisa makanan, kulit buah, daun kering.'
        }
    }
    
    selected_tips = tips_data.get(category, tips_data['Organik'])
    selected_tips['bg_color_start'] = bg_color_start
    selected_tips['bg_color_end'] = bg_color_end
    selected_tips['border_color'] = base_color
    selected_tips['text_color'] = base_color
    
    return selected_tips

# Wrap the entire app content in the main-container for overall layout
with st.container(): # Main container for overall app layout
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # --- Header Section ---
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🌍 EcoSort AI</h1>
        <h2 class="hero-subtitle">
            Klasifikasi Sampah Cerdas dengan Kecerdasan Buatan
            <br>Mari bersama menciptakan lingkungan yang lebih bersih dan berkelanjutan
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # --- Project Explanation ---
    st.markdown("""
    <div class="project-description glass-card">
        <h2>Apa itu EcoSort AI?</h2>
        <p>
            EcoSort AI adalah sebuah aplikasi inovatif yang dirancang untuk membantu Anda mengklasifikasikan berbagai jenis sampah menggunakan teknologi kecerdasan buatan, khususnya model Deep Learning VGG16.
            Dengan mengunggah gambar sampah atau mengambil foto langsung, EcoSort AI akan secara otomatis mengidentifikasi kategori sampah tersebut—apakah itu <b>Organik</b>, <b>Anorganik Daur Ulang</b>, <b>Anorganik Tidak Daur Ulang</b>, atau <b>Bahan Berbahaya dan Beracun (B3)</b>.
            Tujuan utama proyek ini adalah untuk meningkatkan kesadaran akan pentingnya pemilahan sampah yang benar, mendukung praktik daur ulang, dan berkontribusi pada pengelolaan limbah yang lebih efektif demi lingkungan yang lebih hijau dan berkelanjutan.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Upload Section ---
    st.markdown("""
    <div class="glass-card">
        <div class="upload-section">
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Wrap the content and uploader in an st.container
        with st.container(): 
            st.markdown("""
            <div class="upload-card">
                <div class="upload-card-content">
                    <span class="upload-icon">📁</span>
                    <h3 class="upload-title">Upload dari Galeri</h3>
                    <p class="upload-subtitle">Pilih gambar sampah dari perangkat Anda</p>
                </div>
                <div class="button-container">
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Pilih gambar...",
                type=["jpg", "jpeg", "png"],
                key="file_uploader",
                label_visibility="hidden" # Hide default label
            )

    with col2:
        # Wrap the content and camera input/button in an st.container
        with st.container(): 
            st.markdown("""
            <div class="upload-card">
                <div class="upload-card-content">
                    <span class="upload-icon">📸</span>
                    <h3 class="upload-title">Ambil Foto</h3>
                    <p class="upload-subtitle">Gunakan kamera untuk mengambil foto langsung</p>
                </div>
                <div class="button-container">
            """, unsafe_allow_html=True)
            
            # This placeholder will hold either the "Activate Camera" button or the camera input
            camera_col_placeholder = st.empty()

            if st.session_state.show_camera:
                with camera_col_placeholder.container(): # This container ensures camera input stays within the card
                    captured_camera_file = st.camera_input("Ambil foto sampah", key="camera_input", label_visibility="hidden")
                    if captured_camera_file is not None:
                        st.session_state.camera_file_buffer = captured_camera_file # Store captured image
                        st.session_state.show_camera = False # Hide camera after capture
                        st.rerun() # Rerun to process the captured image
            else:
                # Show button if camera is not active
                if camera_col_placeholder.button("📷 Aktifkan Kamera", key="camera_btn", help="Klik untuk mengaktifkan kamera"):
                    st.session_state.show_camera = True
                    st.session_state.camera_file_buffer = None # Clear previous camera image if re-activating
                    st.rerun() # Rerun to show camera input immediately
    st.markdown("</div></div>", unsafe_allow_html=True) # Close upload-section and glass-card

    # --- Choose Source Image ---
    image_source = None
    if uploaded_file is not None:
        image_source = uploaded_file
        # Clear camera buffer if a new file is uploaded
        if st.session_state.camera_file_buffer is not None:
            st.session_state.camera_file_buffer = None
    elif st.session_state.camera_file_buffer is not None:
        image_source = st.session_state.camera_file_buffer
        

    # --- Result Prediction ---
    if image_source is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Loading animation wraps the prediction logic
        with st.spinner("Menganalisis gambar..."):
            time.sleep(1) # Simulate processing time
            predicted_label, confidence, display_img = predict_image(image_source, model, class_labels)
        
        if predicted_label:
            color = category_colors.get(predicted_label, '#6366f1') # Get color based on prediction
            
            # Use Streamlit columns for side-by-side display of image and prediction
            results_col1, results_col2 = st.columns(2)

            with results_col1:
                # Wrap the image in an st.container for styling
                with st.container(): 
                    st.image(image_source, use_column_width=True, caption='Gambar yang Diunggah/Diambil')
            
            with results_col2:
                st.markdown(f"""
                <div class="prediction-card" style="border-left-color: {color};">
                    <h3 class="prediction-title">📊 Hasil Klasifikasi</h3>
                    <div class="prediction-result" style="background: linear-gradient(135deg, {color}15 0%, {color}25 100%); color: {color}; border: 2px solid {color}30;">
                        {predicted_label}
                    </div>
                    <p style="color: #64748b; margin-bottom: 0.5rem;">Tingkat Kepercayaan</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence}%; background: linear-gradient(90deg, {color} 0%, {color}80 100%);"></div>
                    </div>
                    <p style="text-align: center; font-weight: 600; color: {color};">{confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Tips Section with dynamic colors
            tips = get_tips(predicted_label, color)

            st.markdown(f"""
            <div class="tips-section" style="background: linear-gradient(135deg, {tips['bg_color_start']} 0%, {tips['bg_color_end']} 100%); border-left-color: {tips['border_color']};">
                <div class="tips-title" style="color: {tips['text_color']};">
                    <span style="font-size: 1.2rem;">{tips['icon']}</span>
                    {tips['title']}
                </div>
                <div class="tips-content" style="color: {tips['text_color']}e0;">
                    {tips['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.prediction_made = True
        else:
            st.markdown("""
            <div class="custom-alert alert-warning">
                ⚠️ Tidak dapat melakukan klasifikasi. Silakan coba dengan gambar yang lebih jelas.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Welcome message when no image is uploaded
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h2 style="color: #4f46e5; margin-bottom: 1rem;">🚀 Mulai Klasifikasi Sampah</h2>
            <p style="color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">
                Upload gambar dari galeri atau ambil foto langsung menggunakan kamera untuk memulai analisis AI
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 2rem;">
                <div style="padding: 1rem; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 12px;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">♻️</div>
                    <div style="font-weight: 600; color: #0f172a;">Anorganik Daur Ulang</div>
                </div>
                <div style="padding: 1rem; background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border-radius: 12px;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🗑️</div>
                    <div style="font-weight: 600; color: #0f172a;">Anorganik Umum</div>
                </div>
                <div style="padding: 1rem; background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); border-radius: 12px;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚠️</div>
                    <div style="font-weight: 600; color: #0f172a;">Limbah B3</div>
                </div>
                <div style="padding: 1rem; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-radius: 12px;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🌱</div>
                    <div style="font-weight: 600; color: #0f172a;">Organik</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Footer ---
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: rgba(255, 255, 255, 0.8);">
        <p style="margin-bottom: 0.5rem;">Dikembangkan oleh LAI25-SM011</p>
        <p style="font-size: 0.9rem; opacity: 0.7;">Powered by Deep Learning & VGG16 Architecture</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True) # Close main-container div
