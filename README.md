# BERT Sentiment Analysis

Este proyecto implementa un modelo de **BERT** para la clasificación de titulares de noticias en tres categorías: **Bajó**, **Subió**, y **Neutral**. Utiliza la biblioteca **Hugging Face's Transformers** junto con **PyTorch** para el entrenamiento en una GPU **NVIDIA RTX 3070**.

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
