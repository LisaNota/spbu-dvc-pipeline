import os
from catboost import CatBoostClassifier
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# подгружаем параметры
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)
exp_name = params['experiment']['name']


output_dir = f"experiments/experiment_{exp_name}"
os.makedirs(output_dir, exist_ok=True)

data = pd.read_csv('iris.csv')
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1),
                                                    data['target'], test_size=params['train']['test_size'],
                                                    random_state=42)


model = CatBoostClassifier(
    iterations=params['train']['iterations'],
    learning_rate=params['train']['learning_rate'],
    depth=params['train']['depth'],
    random_seed=42,
    loss_function='MultiClass'
)
model.fit(X_train, y_train)
model.save_model(f"{output_dir}/model.cbm")
