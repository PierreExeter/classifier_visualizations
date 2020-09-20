import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier


# CREATE DATASET, SCALE IT and SPLIT INTO TRAIN AND TEST SET
X, y = make_moons(n_samples=100, noise=0.3, random_state=0)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

# DEFINE MESH GRID
h = .02  # step size of the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# DEFINE CLASSIFIER, FIT AND COMPUTE METRICS
name_clf = "KNeighborsClassifier"
clf = KNeighborsClassifier(3)
clf.fit(X_train, y_train)
train_accuracy = round(clf.score(X_train, y_train), 2)
test_accuracy = round(clf.score(X_test, y_test), 2)

# PREDICT PROBABILITY CLASSIFIER
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# DEFINE FIGURE AND COLORMAP
fig, ax = plt.subplots(figsize=(10, 10))
cm = plt.cm.viridis
first, last = cm.colors[0], cm.colors[-1]
cm_bright = ListedColormap([first, last])

# PLOT CONTOUR CLASSIFIER, DECISION BOUNDARY, TRAIN AND TEST DATA
cb = ax.contourf(xx, yy, Z, levels=10, cmap=cm, alpha=0.8)
ax.contour(xx, yy, Z, levels=[.5], colors='k', linestyles='dashed', linewidths=2)
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k', linewidth=2, label="Train data")
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6, label="Test data")

# COLORBAR, AXIS LIMITS, TITLE
fs = 15
cbar = plt.colorbar(cb, ticks=[0, 1])
ax.legend(fontsize=fs, loc='upper left')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel("$X_1$", fontsize=fs)
ax.set_ylabel("$X_2$", fontsize=fs)
ax.set_xticks(())
ax.set_yticks(())
title_str = f"KNeighborsClassifier({clf.n_neighbors}) | Train accuracy: {train_accuracy} | Test accuracy: {test_accuracy}"
ax.set_title(title_str, fontsize=fs)

# SAVE FIGURE
plt.tight_layout()
# plt.show()
plt.savefig('plots/simple_plot_KNN.png', dpi=200, bbox_inches='tight')
