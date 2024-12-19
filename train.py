import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml
import pickle


with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)
exp_name = params['experiment']['name']

output_dir = f"experiments/experiment_{exp_name}"
model_path = f"{output_dir}/model.pth"


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


# Инициализация модели, функции потерь и оптимизатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)

optimizer = optim.SGD(model.parameters(), lr=params['train']['learning_rate'])
loss_function = nn.CrossEntropyLoss()

# Обучение модели
history = []

for epoch in range(params['train']['num_epochs']):
    model.train()
    optimizer.zero_grad()

    inputs = X_train_tensor.to(device)
    labels = y_train_tensor.to(device)

    outputs = model(inputs)
    loss = loss_function(outputs, labels)

    loss.backward()
    optimizer.step()

    # Оценка точности на обучающем наборе
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    history.append(accuracy)


torch.save(model.state_dict(), model_path)
