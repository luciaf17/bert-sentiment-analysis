from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Inicializar la aplicación Flask
app = Flask(__name__)

# Cargar el modelo entrenado y el tokenizador
model = BertForSequenceClassification.from_pretrained('luciaf17/my-bert-sentiment-model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Mover el modelo a la GPU si está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Definir la ruta de la API para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener el texto enviado en la solicitud
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'Falta el parámetro "text" en la solicitud'}), 400
    
    # Tokenizar el texto de entrada
    inputs = tokenizer(data['text'], return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Mover los tensores a la GPU si está disponible
    
    # Hacer la predicción
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    
    # Devolver el resultado como JSON (solo dos clases: Venta y Compra)
    classes = ["Venta", "Compra"]
    result = {'prediction': classes[predicted_class_id]}
    return jsonify(result)

# Iniciar la aplicación
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
