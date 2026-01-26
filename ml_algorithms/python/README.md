# Machine Learning Algorithms (NumPy)

This folder adds classic machine learning algorithms implemented with NumPy to keep the
repository focused on core ML/DL fundamentals.

## Algorithms
- Linear Regression
- Logistic Regression
- Perceptron
- Linear SVM
- Naive Bayes (Gaussian, Multinomial)
- Decision Trees (classification/regression)
- Random Forests (classification/regression)
- K-Nearest Neighbors (classification/regression)
- K-Means clustering
- DBSCAN clustering
- Gaussian Mixture Models (GMM)
- Principal Component Analysis (PCA)
- MLP classifier

## Example
```python
import numpy as np
from linear_models import LogisticRegression
from kmeans import KMeans
from svm import LinearSVM
from mlp import MLPClassifier

x = np.random.randn(200, 4)
labels = (x[:, 0] + x[:, 1] > 0).astype(int)

clf = LogisticRegression(learning_rate=0.1, epochs=500).fit(x, labels)
print(clf.predict(x[:5]))

svm = LinearSVM(learning_rate=0.01, epochs=500).fit(x, labels)
print(svm.predict(x[:5]))

mlp = MLPClassifier(hidden_units=16, epochs=200).fit(x, labels)
print(mlp.predict(x[:5]))

kmeans = KMeans(n_clusters=3, random_state=42).fit(x)
print(kmeans.labels_[:5])
```
