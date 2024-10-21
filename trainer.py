from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# Verificar si hay GPU disponible y asignar el dispositivo adecuado (GPU o CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Usando dispositivo: {device}")

# Cargar los datos
df = pd.read_csv('clean_dataset.csv')

# Dividir los datos en conjuntos de entrenamiento y validación
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Clean_Headline'].tolist(),
    df['Label'].tolist(),
    test_size=0.2,
    random_state=42
)

# Tokenización de los textos
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Crear un dataset personalizado para PyTorch
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # No necesitamos mover a GPU aquí, el `Trainer` lo hará automáticamente.
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Crear datasets de entrenamiento y validación
train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# Cargar el modelo de BERT preentrenado y moverlo a la GPU si está disponible
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# Configurar los argumentos del entrenamiento
training_args = TrainingArguments(
    output_dir='./results',  # Directorio de salida
    num_train_epochs=3,  # Número de épocas de entrenamiento
    per_device_train_batch_size=16,  # Tamaño del batch para entrenamiento
    per_device_eval_batch_size=64,  # Tamaño del batch para evaluación
    warmup_steps=500,  # Pasos de calentamiento
    weight_decay=0.01,  # Decaimiento de peso (regularización)
    evaluation_strategy="epoch",  # Evaluación en cada época
    logging_dir='./logs',  # Directorio para los logs
    logging_steps=10  # Frecuencia de logging
)

# Crear el objeto Trainer
trainer = Trainer(
    model=model,  # Modelo BERT
    args=training_args,  # Argumentos de entrenamiento
    train_dataset=train_dataset,  # Dataset de entrenamiento
    eval_dataset=val_dataset  # Dataset de validación
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo entrenado
trainer.save_model('./results/trained_model')
