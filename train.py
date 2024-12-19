import os
from sklearn.ensemble import RandomForestClassifier
import joblib
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


model = RandomForestClassifier(
    n_estimators=params['train']['n_estimators'],
    max_depth=params['train']['max_depth'],
    random_state=42
)

model.fit(X_train, y_train)
joblib.dump(model, f"{output_dir}/model.pkl")
