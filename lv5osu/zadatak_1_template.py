import matplotlib.colors
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# a)

color = matplotlib.colors.ListedColormap(["blue", "red"])
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=color, label='Training Data', alpha=0.7)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=color, marker='x', label='Testing Data', alpha=0.7)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

# b)
model = LogisticRegression()
model.fit(X_train, y_train)

# c)
color = matplotlib.colors.ListedColormap(["blue", "red"])
print(f"{model.intercept_[0]:.4} + {model.coef_[0, 0]:.4}x_1 + {model.coef_[0, 1]:.4}x_2 = 0")

min = min(X_train[:, 0])
max = max(X_train[:, 0])
xs = np.linspace(min, max, 100)
ys = [(x * model.coef_[0, 0] + model.intercept_[0]) / -model.coef_[0, 1] for x in xs]
plt.scatter(X_train[:, 0], X_train[:, 1], s=5, c=y_train, cmap=color)
plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', c=y_test, cmap=color)
plt.plot(xs, ys)
plt.show()


# d)
y_pred = model.predict(X_test)
print(f" Model accuracy: {accuracy_score(y_test, y_pred):.4}")
print(f"Model precision: {precision_score(y_test, y_pred):.4}")
print(f"   Model recall: {recall_score(y_test, y_pred):.4}")

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
plt.show()

print()

color = matplotlib.colors.ListedColormap(["blue", "red"])
plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', c=y_test == y_pred, cmap=color)
plt.plot(xs, ys)
plt.show()

