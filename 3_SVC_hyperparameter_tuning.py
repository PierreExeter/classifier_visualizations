import numpy as np
import matplotlib.pyplot as plt
import gif
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.svm import SVC


# CREATE DATASET, SCALE IT and SPLIT INTO TRAIN AND TEST SET
X, y = make_moons(n_samples=500, noise=0.3, random_state=0)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

# DEFINE MESH GRID
h = .02  # step size of the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

fs = 15

# WRAP FITTING AND PLOTTING IN A FUNCTION
@gif.frame
def plot_svc(C, C_list=[], train_list=[], test_list=[], max_C=100):
    """ 2D plot of a SVC classifier with a regularization parameter of C """

    # DEFINE CLASSIFIER, FIT AND COMPUTE METRICS
    clf = SVC(gamma=2, C=C, probability=True)
    clf.fit(X_train, y_train)

    # COMPUTE METRICS
    train_accuracy = round(clf.score(X_train, y_train), 2)
    test_accuracy = round(clf.score(X_test, y_test), 2)
    C_list.append(C)
    train_list.append(train_accuracy)
    test_list.append(test_accuracy)

    # PREDICT PROBABILITY CLASSIFIER
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # DEFINE FIGURE AND COLORMAP
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    cm = plt.cm.viridis
    first, last = cm.colors[0], cm.colors[-1]
    cm_bright = ListedColormap([first, last])

    # PLOT CONTOUR CLASSIFIER, DECISION BOUNDARY, TRAIN AND TEST DATA
    cb = ax1.contourf(xx, yy, Z, levels=10, cmap=cm, alpha=0.8)
    ax1.contour(xx, yy, Z, levels=[.5], colors='k', linestyles='dashed', linewidths=2)
    ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                edgecolors='k', marker='o', s=100, linewidth=2, label="Train data")
    ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                edgecolors='k', marker='^', s=100, linewidth=2, label="Test data")

    # PLOT METRICS VS NB NEIGHBORS
    ax2.plot(C_list, train_list, marker='o', markersize=10, color='k', label="Train accuracy")
    ax2.plot(C_list, test_list, marker='^', markersize=10, color='r', label="Test accuracy")

    # COLORBAR, AXIS LIMITS, TITLE
    cbar = plt.colorbar(cb, ticks=[0, 1])
    ax1.legend(fontsize=fs, loc='upper left')
    ax1.set_xlim(xx.min(), xx.max())
    ax1.set_ylim(yy.min(), yy.max())
    ax1.set_xlabel("$X_1$", fontsize=fs)
    ax1.set_ylabel("$X_2$", fontsize=fs)
    ax1.set_xticks(())
    ax1.set_yticks(())

    ax2.set_xlabel("Regularization parameter C", fontsize=fs)
    ax2.set_ylabel("Accuracy value", fontsize=fs)
    ax2.legend(fontsize=fs, loc='lower left')
    ax2.set_xscale('log')
    ax2.set_xlim(10**-2, max_C)
    ax2.set_ylim(0, 1)

    title_str = f"SVC({clf.C}) | Train accuracy: {train_accuracy} | Test accuracy: {test_accuracy}"
    plt.suptitle(title_str, fontsize=fs)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# # Test plot (Comment if not needed, else it will mess up the accuracy plot)
# plot_svc(C=1)
# plt.show()


# LOOP OVER NUMBER OF NEIGHBORS AND SAVE GIF
gif.options.matplotlib['dpi'] = 100

frames = []
C_list = [1/100, 2/100, 3/100, 4/100, 5/100, 1/10, 5/10, 1, 5, 10, 50, 100]

for C in C_list:
    frame = plot_svc(C=C, max_C=100)
    frames.append(frame)

gif.save(frames, "plots/svc_C.gif", duration=1, unit='s')
