# streamlit_app.py
import streamlit as st
import requests

# Título de la app
st.title("Análisis de Sentimiento - Predicción Compra/Venta")

# Ingresar la noticia
text_input = st.text_area("Introduce el texto de la noticia:")

# Botón para hacer la predicción
if st.button("Obtener Predicción"):
    if text_input:
        # Hacer la solicitud a la API Flask
        response = requests.post(
            "https://bert-sentiment-analysis.onrender.com/predict", 
            json={"text": text_input}
        )
        
        if response.status_code == 200:
            prediction = response.json().get("prediction", "Error en la predicción")
            st.success(f"La predicción es: {prediction}")
        else:
            st.error("Error al conectar con la API.")
    else:
        st.warning("Por favor, introduce una noticia.")
