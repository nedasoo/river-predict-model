import time
import pandas as pd
from river import metrics, stream, compose, preprocessing, linear_model, tree


df = pd.read_csv("data/ecommerce_customer_churn_dataset.csv")
model = compose.Pipeline(
    ('one_hot', preprocessing.OneHotEncoder()), 
    ('scale', preprocessing.StandardScaler()),
    #('log_reg', linear_model.LogisticRegression())
    ('tree', tree.HoeffdingTreeClassifier())
)

#Define target to predict
TARGET = 'Churned'
X = df.drop(columns=[TARGET])
y = df[TARGET]

metric = metrics.Accuracy()
print("Running model...")
for i,(x, y) in enumerate(stream.iter_pandas(X, y)):
  y_pred = model.predict_one(x)
  if y_pred is not None:
        metric.update(y, y_pred)

  model.learn_one(x, y)

  if i % 1000 == 0:
    print(f"[{i} samples] Current Accuracy: {metric.get():.2%}")

print(f"Final Accuracy: {metric.get():.2%}")