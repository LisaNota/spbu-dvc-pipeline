"""Модуль для оценки результата модели, создания графиков и сохрания метрик"""

import os
import pandas as pd
import json
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import csv

# Загружаем параметры
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)
exp_name = params['experiment']['name']

# Пути
model_path = f"experiments/experiment_{exp_name}/model.cbm"
output_dir = f"experiments/experiment_{exp_name}"
os.makedirs(output_dir, exist_ok=True)

# Загружаем данные и модель
data = pd.read_csv('iris.csv')

X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1),
                                                    data['target'], test_size=params['train']['test_size'],
                                                    random_state=42)


model = CatBoostClassifier()
model.load_model(model_path)

# Оценка
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Сохранение матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(f"{output_dir}/confusion_matrix.png")


# Сохранение метрик
metrics = classification_report(y_test, y_pred, output_dict=True)


flattened_metrics = []
for label, values in metrics.items():
    if isinstance(values, dict):
        for metric, value in values.items():
            flattened_metrics.append({
                'class': label,
                'metric': metric,
                'value': value
            })
    else:
        flattened_metrics.append({
            'class': label,
            'metric': 'overall',
            'value': values
        })


metrics_path = f"{output_dir}/metrics.csv"
with open(metrics_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['class', 'metric', 'value'])
    writer.writeheader()
    writer.writerows(flattened_metrics)
