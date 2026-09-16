import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de malaria",
    page_icon="🔬"
)

# Título
st.title("🔬 Clasificador de células de malaria")
st.write(
    "Aplicación basada en MobileNetV2 para clasificar imágenes "
    "microscópicas como Parasitized o Uninfected."
)

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("malaria_mobilenetv2_final.keras")

model = cargar_modelo()

# Subir imagen
archivo = st.file_uploader(
    "Selecciona una imagen del conjunto de prueba",
    type=["jpg", "jpeg", "png"]
)

if archivo is not None:

    # Abrir imagen
    imagen = Image.open(archivo).convert("RGB")

    # Mostrar imagen
    st.image(imagen, caption="Imagen seleccionada", width="stretch")

    # Preprocesamiento
    imagen_procesada = imagen.resize((100, 100))
    array_imagen = np.array(imagen_procesada, dtype=np.float32)

    # Agregar dimensión del batch
    array_imagen = np.expand_dims(array_imagen, axis=0)

    # Predicción
    probabilidad = model.predict(array_imagen, verbose=0)[0][0]

    # Interpretar resultado
    if probabilidad >= 0.5:
        clase = "Parasitized"
    else:
        clase = "Uninfected"

    st.subheader("Resultado de la predicción")
    st.write(f"**Clase predicha:** {clase}")
    st.write(f"**Probabilidad de Parasitized:** {probabilidad:.2%}")
