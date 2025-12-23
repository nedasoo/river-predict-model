import pandas as pd
from river import metrics, stream, compose, preprocessing, forest
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

df = pd.read_csv("data/ecommerce_customer_churn_dataset.csv")
TARGET = 'Churned'
X = df.drop(columns=[TARGET])
y = df[TARGET]

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

def get_fresh_model():
    return compose.Pipeline(
        ('features', compose.TransformerUnion(
            ('numeric', compose.Pipeline(
                compose.Select(*numerical_features),
                preprocessing.StatImputer(),
            )),
            ('categorical', compose.Pipeline(
                compose.Select(*categorical_features),
                preprocessing.StatImputer(),
                preprocessing.OneHotEncoder()
            ))
        )),
        ('classifier', forest.ARFClassifier(
            n_models=100, 
            seed=42, 
            max_depth=20, 
            leaf_prediction='nba', 
            grace_period=500, 
            lambda_value=1
        ))
    )

def run_experiment(train_limit):
    model = get_fresh_model()
    test_metric = metrics.Accuracy()
    
    print(f"--- Training on {train_limit} rows ---")
    for i, (x, y_true) in enumerate(stream.iter_pandas(X, y), 1):
        model.learn_one(x, y_true)
        if i >= train_limit:
            break
            
    print(f"Testing against all {len(df)} rows...")
    for i, (x, y_true) in enumerate(stream.iter_pandas(X, y)):
        y_pred = model.predict_one(x)
        if y_pred is not None and i >= 2500:
            test_metric.update(y_true, y_pred)
            
    return train_limit, test_metric.get()

if __name__ == '__main__':
  checkpoints = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]

  results_list = []
  with ProcessPoolExecutor() as executor:
    #run on different cores
    results_list = list(executor.map(run_experiment, checkpoints))

  results = dict(sorted(results_list))

  plt.figure(figsize=(10, 6))
  plt.plot(list(results.keys()), list(results.values()), marker='s', color='teal')
  plt.title("Parallel Experiment Results: Accuracy vs. Training Size")
  plt.xlabel("Samples Used for Training")
  plt.ylabel("Accuracy")
  plt.grid(True)
  plt.show()