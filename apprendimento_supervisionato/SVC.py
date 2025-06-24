import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
from sklearn.model_selection import GridSearchCV 
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from models_results import write_results

warnings.filterwarnings('ignore')

df = pd.read_csv('./apprendimento_supervisionato/dataset_model.csv')

# Separazione features e target
X = df.drop(['Depression'], axis=1)
y = df['Depression']

# Split del dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Peso delle classi per la variabile target
class_weight={0: 2, 1: 5}

svc=SVC(class_weight=class_weight, random_state=42) 

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
print('Model recall score with default hyperparameters: {0:0.4f}'. format(recall_score(y_test, y_pred)))

param_grid = {
    'C': [10], 
	'kernel': ['rbf'],
    'gamma': ['scale'],
    'decision_function_shape': ['ovo']
} 

# GridSearch
grid = GridSearchCV(SVC(class_weight=class_weight, random_state=42), param_grid, cv=10, scoring='accuracy') 
 
grid.fit(X_train, y_train)

print(grid.best_params_) 
 
print(grid.best_estimator_)

grid_predictions = grid.predict(X_test) 

print('Model accuracy score after grid search: {0:0.4f}'. format(accuracy_score(y_test, grid_predictions)))
print('Model recall score with after grid search: {0:0.4f}'. format(recall_score(y_test, grid_predictions)))

accuracy = round(accuracy_score(y_test, grid_predictions), 3)
recall = round(recall_score(y_test, grid_predictions), 3)

print(classification_report(y_test, grid_predictions))

# Matrice di confusione
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', square=True)
plt.title('Matrice di Confusione')
plt.xlabel('Predetto')
plt.ylabel('Reale')
plt.savefig(f'./apprendimento_supervisionato/results/SVM_matrix.png', bbox_inches='tight')

write_results("./apprendimento_supervisionato/accuracy_recall.txt", "SVC", accuracy, recall)