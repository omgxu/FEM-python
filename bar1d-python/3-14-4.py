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

# 计算 L2 范数误差
def compute_l2_error(n):
    x_fine = np.linspace(0, 1, 1000)  # 高分辨率采样点
    points = [(i / n, f(i / n)) for i in range(n + 1)]  # 插值点
    y_actual = f(x_fine)
    y_approx = [linear_approximation(xi, points) for xi in x_fine]
    error_squared = (y_actual - y_approx)**2
    l2_error = np.sqrt(np.trapz(error_squared, x_fine))  # 数值积分计算 L2 范数误差
    return l2_error

# 绘制 e_L2-h 双对数曲线
def plot_l2_error():
    ns = [2, 4, 8, 16, 32, 64]  # 不同划分数量
    hs = [1 / n for n in ns]  # 单元长度 h = 1/n
    l2_errors = [compute_l2_error(n) for n in ns]  # 计算对应的 L2 范数误差

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(hs, l2_errors, marker="o", label="e_L2 vs h")  # 双对数曲线
    ax.set_title("L2 Error vs. Cell Size (h)")
    ax.set_xlabel("Cell Size (h)")
    ax.set_ylabel("L2 Error (e_L2)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    plt.show()

# 主程序运行
if __name__ == "__main__":
    print("Plotting e_L2-h double-logarithmic curve...")
    plot_l2_error()