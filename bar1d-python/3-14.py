import numpy as np
import matplotlib.pyplot as plt

# 定义函数 y = x^3 和其导数 y' = 3x^2
def f(x):
    return x**3

def df(x):
    return 3 * x**2

# 定义线性单元插值函数
def linear_approximation(x, points):
    """
    根据给定点进行线性插值
    :param x: 需要插值的点
    :param points: 插值点 [(x0, y0), (x1, y1), ...]
    :return: 插值结果
    """
    for i in range(len(points) - 1):
        if points[i][0] <= x <= points[i + 1][0]:
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None

# 绘制 y = x^3 及其线性单元近似
def plot_function_and_approximations():
    x = np.linspace(0, 1, 500)  # 实际函数的高分辨率采样
    y = f(x)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, label="y = x^3", color="blue")

    # 划分不同数量的线性单元并绘制近似曲线
    for n in [2, 4, 8]:
        points = [(i / n, f(i / n)) for i in range(n + 1)]  # 插值点
        x_approx = np.linspace(0, 1, 500)
        y_approx = [linear_approximation(xi, points) for xi in x_approx]
        ax.plot(x_approx, y_approx, label=f"Linear Approximation (n={n})", linestyle="--")

    ax.set_title("Function y = x^3 and Linear Approximations")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.grid()
    plt.show()

# 绘制 y = x^3 的导数及其线性单元的导数
def plot_derivative_and_approximations():
    x = np.linspace(0, 1, 500)  # 实际导数的高分辨率采样
    y = df(x)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, label="Derivative y' = 3x^2", color="blue")

    # 划分不同数量的线性单元并绘制导数的近似曲线
    for n in [2, 4, 8]:
        points = [(i / n, f(i / n)) for i in range(n + 1)]  # 插值点
        slopes = [(points[i + 1][1] - points[i][1]) / (points[i + 1][0] - points[i][0]) for i in range(n)]
        
        # 在每个区间内绘制导数的常数值
        for i in range(n):
            x_segment = np.linspace(points[i][0], points[i + 1][0], 500)
            y_segment = slopes[i] * np.ones_like(x_segment)
            ax.plot(x_segment, y_segment, label=f"Slope (n={n})" if i == 0 else None, linestyle="--")

    ax.set_title("Derivative y' = 3x^2 and Linear Approximations")
    ax.set_xlabel("x")
    ax.set_ylabel("y'")
    ax.legend()
    plt.grid()
    plt.show()

# 主程序运行
if __name__ == "__main__":
    print("Plotting y = x^3 and its linear approximations...")
    plot_function_and_approximations()

    print("Plotting the derivative of y = x^3 and its linear approximations...")
    plot_derivative_and_approximations()