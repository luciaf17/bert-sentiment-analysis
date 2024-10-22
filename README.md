# BERT Sentiment Analysis

Este proyecto implementa un modelo de **BERT** para la clasificación de titulares de noticias en dos categorías: **Bajó** y **Subió**. Utiliza la biblioteca **Hugging Face's Transformers** junto con **PyTorch** para el entrenamiento. Somos integrantes del grupo 8 del curso IA para Tecnologas con experiencia.

La idea es crear un indicador de Compra/Venta para el trading de acciones del mercado estadounidense.

## Instalar las dependencias:
Este proyecto utiliza PyTorch, transformers y otras librerías de NLP y machine learning. Todas las dependencias necesarias están listadas en el archivo requirements.txt.

## Instalación

### 1. Clonar el repositorio:

`git clone https://github.com/luciaf17/bert-sentiment-analysis.git
cd bert-sentiment-analysis`


### 2. Crear y activar un entorno virtual:
Si no tienes un entorno virtual creado, sigue estos pasos para crear uno en la carpeta del proyecto:

En Windows:

`python -m venv bertvenv`

`.\bertvenv\Scripts\activate`

### 3. Instalar las dependencias:

`pip install -r requirements.txt`


## Entrenar el modelo

### 1. Entrenamiento:
Para entrenar el modelo BERT en los datos de titulares de noticias, simplemente ejecuta el archivo trainer.py

Esto entrenará el modelo en el dataset limpio proporcionado y guardará los resultados en la carpeta results

### 2. Evaluar el modelo:

Para evaluar el modelo entrenado en el conjunto de validación, puedes ejecutar el archivo evaluate.py

## Resultados

Los resultados del modelo entrenado se guardarán en la carpeta results/, incluyendo los checkpoints del modelo y cualquier métrica de rendimiento obtenida durante la evaluación.


## Disponibilidad del modelo

Se sube a Hugging Face para luego desplegarlo en Render app.

En este colab, podemos llamar a la API enviandole la noticia, y la misma devuelve una respuesta que funciona como indicador de: Compra o Venta.

https://colab.research.google.com/drive/10Q0iUZS5NAFr19sH1kJnlVyW2ED9zEmD#scrollTo=WVWcZrC_50Xa

# API de Predicción de Sentimientos Financieros

Esta API permite realizar predicciones de sentimiento financiero (Compra/Venta) utilizando un modelo de clasificación basado en BERT. La API toma un texto (por ejemplo, un titular de noticia financiera en ingles) y devuelve si sugiere una acción de "Compra" o "Venta".

## Características del modelo:

Modelo utilizado: `bert-base-uncased`.
Entrenamiento sobre un dataset financiero especializado.
Clasificación en dos categorías: Compra y Venta.

### Cómo funciona:

La API está diseñada para recibir una solicitud POST con un texto, el cual es tokenizado y procesado por el modelo para devolver una predicción sobre el sentimiento de la noticia.

## Frontend de ejemplo

Puedes ver como funcionan las predicciones en la siguiente URL `https://bert-sentiment-analysis.streamlit.app/`

### Ejemplo de Uso:

-Endpoint: `/predict`

-Método HTTP: `POST`

-Formato de entrada: `JSON`

-URL: `"https://bert-sentiment-analysis.onrender.com/predict"`

-El parámetro clave debe ser text, que contiene el texto (por ejemplo, una noticia o titular en idioma INGLES).

-Respuesta: Un JSON que contiene la predicción entre "Compra" o "Venta".

-Solicitud: Envía una solicitud POST al endpoint `/predict` con el siguiente formato:

```json
{
  "text": "China announces New intervention in Taiwan"
}

Respuesta:

json
{
  "prediction": "Compra"
}

Si se desea trabajar de manera local, es necesario cambiar en el archivo de streamlit_app.py por la url local: `"http://127.0.0.1:5000/predict"`.
Los comandos para ejecutar la API y el Front so los siguientes:

`py app.py`

`streamlit run streamlit_app.py`