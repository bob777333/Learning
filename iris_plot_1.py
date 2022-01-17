import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

#========= load iris data set
iris = datasets.load_iris()

#========= check the iris data set features
print(iris)
#print(iris.data)
print(iris['DESCR'])

#=======Use two first features
X = iris.data[:, :2]  # we only take the first two features.

y = iris.target

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# Plot 3-D
fig = plt.figure(1, figsize=(8, 6))
ax_3D = Axes3D(fig, elev=-150, azim=110, auto_add_to_figure=False)
fig.add_axes(ax_3D)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax_3D.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=y,
    cmap=plt.cm.Set1,
    edgecolor="k",
    s=40,
)
ax_3D.set_title("First three PCA directions")
ax_3D.set_xlabel("1st eigenvector")
ax_3D.w_xaxis.set_ticklabels([])
ax_3D.set_ylabel("2nd eigenvector")
ax_3D.w_yaxis.set_ticklabels([])
ax_3D.set_zlabel("3rd eigenvector")
ax_3D.w_zaxis.set_ticklabels([])

plt.show()