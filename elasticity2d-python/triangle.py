import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# 材料参数和几何定义
E = 30e6  # 弹性模量 (Pa)
nu = 0.3   # 泊松比
t = 1    # 单元厚度 (m)

# 用户输入四边形的四个顶点坐标
def get_quadrilateral():
    print("请输入四边形的四个顶点坐标 (x, y)，按逆时针顺序输入:")
    nodes = []
    for i in range(4):
        x = float(input(f"输入第 {i+1} 个顶点的 x 坐标: "))
        y = float(input(f"输入第 {i+1} 个顶点的 y 坐标: "))
        nodes.append([x, y])
    return np.array(nodes)

# 生成四边形区域内的密集节点
def generate_dense_nodes(quadrilateral, num_divisions):
    """
    在四边形区域内生成密集节点。
    :param quadrilateral: 四边形的四个顶点坐标 (形状为 4x2 的数组)。
    :param num_divisions: 每条边划分的段数。
    :return: 密集节点数组。
    """
    # 提取四边形的四个顶点
    x_min, y_min = quadrilateral.min(axis=0)
    x_max, y_max = quadrilateral.max(axis=0)

    # 在区域内生成均匀分布的节点
    x = np.linspace(x_min, x_max, num_divisions + 1)
    y = np.linspace(y_min, y_max, num_divisions + 1)
    xv, yv = np.meshgrid(x, y)
    dense_nodes = np.vstack((xv.ravel(), yv.ravel())).T

    # 过滤掉不在四边形内部的点
    hull = Delaunay(quadrilateral)
    dense_nodes = dense_nodes[hull.find_simplex(dense_nodes) >= 0]

    return dense_nodes

# 计算平面应力问题的材料矩阵D
def plane_stress_material_matrix(E, nu):
    D = (E / (1 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])
    return D

# 计算三角形单元的B矩阵
def compute_B_matrix(nodes, element):
    # 提取单元节点坐标
    x1, y1 = nodes[element[0]]
    x2, y2 = nodes[element[1]]
    x3, y3 = nodes[element[2]]

    # 面积计算
    A = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

    # 形函数梯度
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    # B矩阵
    B = (1 / (2 * A)) * np.array([
        [b1, 0, b2, 0, b3, 0],
        [0, c1, 0, c2, 0, c3],
        [c1, b1, c2, b2, c3, b3]
    ])
    return B, A

# 主程序
def main():
    # 获取四边形的节点坐标
    quadrilateral = get_quadrilateral()

    # 生成密集节点
    num_divisions = 10  # 每条边划分的段数
    nodes = generate_dense_nodes(quadrilateral, num_divisions)

    # 使用 Delaunay 三角化方法划分网格
    tri = Delaunay(nodes)
    elements = tri.simplices  # 单元连接关系 (三角形单元)

    num_nodes = len(nodes)
    num_dofs = 2 * num_nodes  # 每个节点有2个自由度 (u, v)

    # 全局刚度矩阵和力向量
    K_global = np.zeros((num_dofs, num_dofs))
    F_global = np.zeros(num_dofs)

    # 材料矩阵D
    D = plane_stress_material_matrix(E, nu)

    # 组装全局刚度矩阵
    for element in elements:
        B, A = compute_B_matrix(nodes, element)
        k_element = t * A * np.dot(B.T, np.dot(D, B))  # 单元刚度矩阵

        # 将单元刚度矩阵组装到全局刚度矩阵
        dof_indices = np.concatenate([element * 2, element * 2 + 1])
        for i, ii in enumerate(dof_indices):
            for j, jj in enumerate(dof_indices):
                K_global[ii, jj] += k_element[i, j]

    # 施加边界条件（固定四边形左边的所有节点）
    fixed_nodes = np.where(nodes[:, 0] == nodes[:, 0].min())[0]  # 固定左边节点
    for node in fixed_nodes:
        dof_u = 2 * node      # u 自由度索引
        dof_v = 2 * node + 1  # v 自由度索引
        K_global[dof_u, :] = 0  # 清空 u 自由度对应的行
        K_global[dof_v, :] = 0  # 清空 v 自由度对应的行
        K_global[dof_u, dof_u] = 1  # 将对角线元素设为 1
        K_global[dof_v, dof_v] = 1  # 将对角线元素设为 1
        F_global[dof_u] = 0  # 将外力向量中对应 u 的值设为 0
        F_global[dof_v] = 0  # 将外力向量中对应 v 的值设为 0

    # 在指定边上施加力
    edge_index = int(input("选择要施加力的边 (0: 左边, 1: 上边, 2: 右边, 3: 下边): "))
    total_force = float(input("输入总力大小 (N): "))

    # 确定目标边的节点
    if edge_index == 0:  # 左边
        target_nodes = np.where(nodes[:, 0] == nodes[:, 0].min())[0]
    elif edge_index == 1:  # 上边
        target_nodes = np.where(nodes[:, 1] == nodes[:, 1].max())[0]
    elif edge_index == 2:  # 右边
        target_nodes = np.where(nodes[:, 0] == nodes[:, 0].max())[0]
    elif edge_index == 3:  # 下边
        target_nodes = np.where(nodes[:, 1] == nodes[:, 1].min())[0]
    else:
        raise ValueError("无效的边索引")

    # 分配力到目标节点
    force_per_node = total_force / len(target_nodes)
    forces = np.zeros((num_nodes, 2))
    if edge_index in [1, 3]:  # 上边或下边，力方向向下
        forces[target_nodes, 1] = -force_per_node
    elif edge_index in [0, 2]:  # 左边或右边，力方向向右
        forces[target_nodes, 0] = force_per_node

    # 将力加载到全局力向量
    for i, force in enumerate(forces):
        F_global[2*i:2*i+2] = force

    # 求解线性方程组
    U = np.linalg.solve(K_global, F_global)

    # 输出结果
    print("节点位移 (u, v):")
    for i in range(num_nodes):
        print(f"节点 {i}: u = {U[2*i]:.6e}, v = {U[2*i+1]:.6e}")

    # 变形后的节点坐标
    deformed_nodes = nodes.copy()
    displacement_scale = 1000  # 放大位移的比例
    for i in range(num_nodes):
        deformed_nodes[i, 0] += displacement_scale * U[2*i]
        deformed_nodes[i, 1] += displacement_scale * U[2*i+1]

    # 绘制原始网格和变形后的网格
    plt.figure(figsize=(10, 5))

    # 原始网格
    plt.subplot(1, 2, 1)
    plt.triplot(nodes[:, 0], nodes[:, 1], elements, color='black')
    plt.plot(nodes[:, 0], nodes[:, 1], 'o', label="Nodes")
    plt.title("Original Mesh")
    plt.axis('equal')

    # 变形后的网格
    plt.subplot(1, 2, 2)
    plt.triplot(deformed_nodes[:, 0], deformed_nodes[:, 1], elements, color='red')
    plt.plot(deformed_nodes[:, 0], deformed_nodes[:, 1], 'o', label="Deformed Nodes")
    plt.title(f"Deformed Mesh (Scaled by {displacement_scale})")
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

    # 应力计算和可视化
    stress_results = []
    for element in elements:
        B, A = compute_B_matrix(nodes, element)
        u_element = U[np.concatenate([element * 2, element * 2 + 1])]
        strain = np.dot(B, u_element)
        stress = np.dot(D, strain)
        stress_results.append(stress)

    # 提取正应力和剪应力
    sigma_x = [stress[0] for stress in stress_results]
    sigma_y = [stress[1] for stress in stress_results]
    tau_xy = [stress[2] for stress in stress_results]

    # 绘制应力分布图
    plt.figure(figsize=(15, 5))

    # 正应力 σ_x
    plt.subplot(1, 3, 1)
    plt.tripcolor(nodes[:, 0], nodes[:, 1], elements, facecolors=sigma_x, cmap='viridis', edgecolors='k')
    plt.title("σ_x Stress")
    plt.colorbar(label="Stress (Pa)")
    plt.axis('equal')

    # 正应力 σ_y
    plt.subplot(1, 3, 2)
    plt.tripcolor(nodes[:, 0], nodes[:, 1], elements, facecolors=sigma_y, cmap='viridis', edgecolors='k')
    plt.title("σ_y Stress")
    plt.colorbar(label="Stress (Pa)")
    plt.axis('equal')

    # 剪应力 τ_xy
    plt.subplot(1, 3, 3)
    plt.tripcolor(nodes[:, 0], nodes[:, 1], elements, facecolors=tau_xy, cmap='viridis', edgecolors='k')
    plt.title("τ_xy Stress")
    plt.colorbar(label="Stress (Pa)")
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()