import pandas as pd
from river import metrics, stream, compose, preprocessing, forest, utils
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/ecommerce_customer_churn_dataset.csv")
TARGET = 'Churned'
X = df.drop(columns=[TARGET])
y = df[TARGET]
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

model = compose.Pipeline(
    ('features', compose.TransformerUnion(
        ('numeric', compose.Pipeline(
            compose.Select(*numerical_features),
            preprocessing.StandardScaler(),
            preprocessing.StatImputer(),
        )),
        ('categorical', compose.Pipeline(
            compose.Select(*categorical_features),
            preprocessing.StatImputer(),
            preprocessing.OneHotEncoder()
        ))
    )),
    ('classifier', forest.ARFClassifier(n_models=200, seed=42, max_depth=20, leaf_prediction='nba', grace_period=500, lambda_value=1))
)

metric_acc = metrics.Accuracy()
metric_auc = metrics.ROCAUC()
metric_f1 = metrics.MacroF1()
rolling_acc = utils.Rolling(metrics.Accuracy(), window_size=1000)
cm = metrics.ConfusionMatrix()

for i, (x, y_true) in enumerate(stream.iter_pandas(X, y), 1):
  y_pred = model.predict_one(x)
  y_prob = model.predict_proba_one(x)
  
  if i > 5000 and y_pred is not None:
      metric_acc.update(y_true, y_pred)
      metric_auc.update(y_true, y_prob)
      rolling_acc.update(y_true, y_pred)
      metric_f1.update(y_true, y_pred)
      cm.update(y_true, y_pred)
  
  # Learn
  model.learn_one(x, y_true)
  
  if i % 1000 == 0:
    print(f"[{i} samples] Acc: {metric_acc.get():.2%}, ROC-AUC: {metric_auc.get():.2%}, Rolling Acc: {rolling_acc.get():.2%}")

print(f"\nFinal Accuracy: {metric_acc.get():.2%}")
print(f"Final ROC-AUC: {metric_auc.get():.2%}")
print(f"Final Macro F1: {metric_f1.get():.2%}")

cm_df = pd.DataFrame(cm.data).fillna(0).astype(int)
cm_df = cm_df.sort_index(axis=0).sort_index(axis=1)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title(f'Final Confusion Matrix\nAccuracy: {metric_acc.get():.2%}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
