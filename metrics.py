import os
import torch
import torch.nn as nn
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import csv


with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)
exp_name = params['experiment']['name']

model_path = f"experiments/experiment_{exp_name}/model.pth"
output_dir = f"experiments/experiment_{exp_name}"

data = pd.read_csv('iris.csv', index_col=0)
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('target', axis=1),
    data['target'],
    test_size=params['train']['test_size'],
    random_state=params['train']['random_seed']
)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # 4 - количество признаков
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)   # 3 — кол-во классов

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = MLP().to(device)  # Инициализация модели
model.load_state_dict(torch.load(model_path))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# Оценка модели
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    _, y_pred = torch.max(y_pred_tensor, 1)

y_true = y_test_tensor.cpu().numpy()
y_pred = y_pred.cpu().numpy()

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(f"{output_dir}/confusion_matrix.png")
plt.close()

metrics = classification_report(y_true, y_pred, output_dict=True)
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
