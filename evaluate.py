from transformers import BertTokenizer, BertForSequenceClassification, Trainer
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Cargar los datos
df = pd.read_csv('clean_dataset.csv')

# Dividir los datos en conjuntos de validación
_, val_texts, _, val_labels = train_test_split(
    df['Clean_Headline'].tolist(),
    df['Label'].tolist(),
    test_size=0.2,
    random_state=42
)

# Tokenización de los textos
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Crear un dataset personalizado para PyTorch
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

val_dataset = SentimentDataset(val_encodings, val_labels)

# Cargar el modelo previamente entrenado
model = BertForSequenceClassification.from_pretrained('./results/trained_model', num_labels=2, ignore_mismatched_sizes=True)

# Crear el objeto Trainer (solo para evaluación)
trainer = Trainer(
    model=model,
    eval_dataset=val_dataset
)

# Evaluar el modelo
trainer.evaluate()

# Generar predicciones sobre el conjunto de validación
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=1)

# Mostrar el informe de clasificación
print(classification_report(val_labels, preds, target_names=["Bajó", "Subió"]))
