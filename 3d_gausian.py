import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr

mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T


def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z, title):
    """
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    :param title: The title that will appear above the plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.title(title)
    plt.show()


def plot_2d(x_y, title):
    """
    plot points in 2D
    :param x_y: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # Question 11:
    plot_3d(x_y_z, 'Q11: Random points')

    # Question 12:
    s = np.array([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 2]])
    data = np.matmul(s, x_y_z)
    plot_3d(data, 'Q12: Data transformed by scaling matrix')

    # Question 13:
    rand_mat = get_orthogonal_matrix(3)
    data = np.matmul(rand_mat, data)
    plot_3d(data, 'Q13: Scaled data multiplied by random orthogonal matrix')

    # Question 14:
    plot_2d(x_y_z, 'Q14: Projection to x,y axes')

    # Question 15:

    # Question 16:
