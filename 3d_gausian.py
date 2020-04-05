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
    :param title: The title that will appear above the plot
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


def q11():
    plot_3d(x_y_z, 'Q11: Random points')


def q12():
    s = np.array([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 2]])
    data = np.matmul(s, x_y_z)
    plot_3d(data, 'Q12: Data transformed by scaling matrix')
    return data


def q13(data):
    rand_mat = get_orthogonal_matrix(3)
    data = np.matmul(rand_mat, data)
    plot_3d(data, 'Q13: Scaled data multiplied by random orthogonal matrix')


def q14():
    plot_2d(x_y_z, 'Q14: Marginal distribution - Projection to x,y axes')


def q15():
    bad_points = []
    for i in range(len(x_y_z[2])):
        if 0.1 <= x_y_z[2][i] or x_y_z[2][i] <= -0.4:
            bad_points.append(i)
    data = np.delete(x_y_z, bad_points, axis=1)
    plot_2d(data, 'Q15: Conditional distribution - Projection of points to the x, y axes')


def q16():
    data = np.random.binomial(1, 0.25, (100000, 1000))
    epsilon = [0.25]
    sub_question_a(data)
    sub_questions_b_c(data, epsilon)


def sub_question_a(data):
    first_five_estimated_means = []
    colors = ['red', 'blue', 'green', 'yellow', 'purple']
    for i in range(5):
        first_five_estimated_means.append(np.cumsum(data[i]) / np.arange(1, 1001))
    for i in range(5):
        plt.plot(np.arange(1, 1001),
                 first_five_estimated_means[i],
                 color=colors[i],
                 label=f'Estimated mean of sequence no. {i}')
    plt.legend(loc='best', shadow=True, fontsize='small')
    plt.xlabel('m - Number of samples')
    plt.ylabel('Estimated means')
    plt.title('Q16 (a): Estimated means of sequences')
    plt.show()


def sub_questions_b_c(data, epsilon):
    means = np.array([np.cumsum(line) / np.arange(1, 1001) for line in data])
    transposed_means = means.transpose()
    for i, eps in enumerate(epsilon):
        plt.plot(np.arange(1, 1001),
                 (np.repeat(1, 1000) / np.cumsum(np.repeat(4 * eps ** 2, 1000))).clip(max=1),
                 color='red',
                 label=f'Chebyshev bound')
        plt.plot(np.arange(1, 1001),
                 (np.repeat(2, 1000) * np.exp(np.cumsum(np.repeat(-2 * eps ** 2, 1000)))).clip(max=1),
                 color='blue',
                 label=f'Hoeffding bound')
        percentage = [len(list(filter(lambda x: abs(x - 0.25) >= eps,
                                      transposed_means[j]))) / len(means) for j in range(1000)]
        plt.plot(np.arange(1, 1001), percentage, color='green', label='Percentage of samples satisfying given event')
        plt.title(f'Q16 (b, c): Bounds for ${{\\varepsilon}}$ = {eps}')
        plt.xlabel('m - Number of samples')
        plt.ylabel('Bounds')
        plt.legend(loc='best', shadow=True, fontsize='small')
        plt.show()


if __name__ == '__main__':
    # Question 11:
    q11()

    # Question 12:
    scaled_data = q12()

    # Question 13:
    q13(scaled_data)

    # Question 14:
    q14()

    # Question 15:
    q15()

    # Question 16:
    q16()
