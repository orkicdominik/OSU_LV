import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# K = 5
knn_model5 = KNeighborsClassifier(n_neighbors=5)
knn_model5.fit(X_train_n, y_train)
y_train_knn5 = knn_model5.predict(X_train_n)
y_test_knn5 = knn_model5.predict(X_test_n)


# K = 1
knn_model1 = KNeighborsClassifier(n_neighbors=1)
knn_model1.fit(X_train_n, y_train)
y_train_knn1 = knn_model1.predict(X_train_n)
y_test_knn1 = knn_model1.predict(X_test_n)


# K = 100
knn_model100 = KNeighborsClassifier(n_neighbors=5)
knn_model100.fit(X_train_n, y_train)
y_train_knn100 = knn_model100.predict(X_train_n)
y_test_knn100 = knn_model100.predict(X_test_n)




# Vizualizacija granice odluke
plot_decision_regions(X_train_n, y_train, classifier=knn_model5)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("K=5, Tocnost: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn5)))
plt.tight_layout()
plt.show()


# Vizualizacija granice odluke
plot_decision_regions(X_train_n, y_train, classifier=knn_model1)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("K=1, Tocnost: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn1)))
plt.tight_layout()
plt.show()

# Vizualizacija granice odluke
plot_decision_regions(X_train_n, y_train, classifier=knn_model100)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("K=100, Tocnost: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn100)))
plt.tight_layout()
plt.show()


param_grid = {'n_neighbors': np.arange(1, 51)}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_n, y_train)
print("\nOptimalna vrijednost:", grid_search.best_params_['n_neighbors'])

# SVM s RBF kernelom, primjer s C=1 i gamma=0.1
svm_model = svm.SVC(kernel='rbf', C=100.0, gamma=10.0)
svm_model.fit(X_train_n, y_train)

# Predikcija i točnost
y_test_svm = svm_model.predict(X_test_n)
print("\nSVM (RBF kernel, C=100.0, gamma=10.0):")
print("Tocnost test: {:.3f}".format(accuracy_score(y_test, y_test_svm)))

# Prikaz granice odluke
plot_decision_regions(X_train_n, y_train, classifier=svm_model)
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.title("SVM (RBF kernel, C=100.0, gamma=10.0)")
plt.tight_layout()
plt.show()

