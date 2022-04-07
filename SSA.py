from statistics import mean

import numpy as np


def seq2trajectory(seq, window_size=None):
    """
    :author: 高世平 2021-07-15
    :param seq: np.ndarray 任意长度的一维向量
    :param window_size: integer 从seq中按照固定的长度一次取出其中的等长子序列
    :return: trajectory_matrix np.ndarray (window_size, K) 轨迹矩阵
            [
                [x1, x2, x3,...... , xk  ],
                [x2, x3, x4,...... , xk+1],
                [x3, x4, x5,...... , xk+2],
                [........................],
                [xL, xL+1,xL+2.........,XN],
            ]
            其中L=window_size, N=len(seq), K=len(seq)-window_size+1
    """
    if window_size is None:
        window_size = len(seq) // 2

    N = len(seq)
    K = N - window_size + 1
    return np.array([seq[i: i + K] for i in range(window_size)])


def one2multi(trajectory_matrix, top_k_component=None):
    """
    :author: 高世平 2021-07-15
    :func:将轨迹矩阵由svd方法分解为多个矩阵之和
    :param trajectory_matrix: np.ndarray (L, K) 轨迹矩阵
    :param top_k_component: integer 返回矩阵分解的分量数量，默认为输入轨迹矩阵的秩数
    :return: component_list : list of np.ndarray (L, K), 列表长度为rank_len，列表中矩阵之和即为轨迹矩阵

    """
    if top_k_component is None:
        top_k_component = np.linalg.matrix_rank(trajectory_matrix)

    # 奇异值分解，U(L,L), sigma(min(L,K),), VT(K,K)
    U, sigma, VT = np.linalg.svd(trajectory_matrix)

    # 矩阵分量列表
    component_list = []
    for i in range(top_k_component):
        # 矩阵分解后，每个分量为sigma[i](该分量系数) 乘以svd分解得的该分量的行向量与列向量乘积
        single_component = sigma[i] * U[:, i][:, None] @ VT[i, :][:, None].T
        component_list.append(single_component)

    return component_list


def average_anti_diag(X):
    """
    :author: 高世平 2021-07-15
    :func:计算矩阵反对角线的均值
    :param X: np.ndarray (L, K) 一个二维矩阵
    :return: anti_diag_mean: np.ndarray (L+K-1, ) average_anti_diag 输入矩阵的各个反对角线上元素的平均值
    """
    offset_list = range(-X.shape[0] + 1, X.shape[1])
    anti_diag = []
    for offset in offset_list:
        anti_diag.append(X[::-1, :].diagonal(offset))
    anti_diag_mean = np.array([mean(x) for x in anti_diag])

    return anti_diag_mean


def ssa(seq, window_size=None, top_k_component=None):
    """
    :author: 高世平 2021-07-16
    :func: 将输入的任意长度向量使用ssa方法去除噪声
    :param seq: np.ndarray 任意长度的一维向量
    :param window_size: 结果的稳定性，该值越大，计算量越大，但结果更稳定
    :param top_k_component: 去噪程度，该值越小，去噪程度越高
    :return: 经过ssa去噪后的结果
    """
    # 第一步：嵌入，将seq 转化为轨迹矩阵
    trajectory = seq2trajectory(seq, window_size=window_size)

    # 第二步：分解，将轨迹矩阵分解为多个分量
    multi_component_list = one2multi(trajectory_matrix=trajectory, top_k_component=top_k_component)

    # 第三步：重构轨迹矩阵，将其中多个分量相加，重构矩阵
    new_trajectory_matrix = np.array(multi_component_list).sum(0)

    # 第四步：重构序列
    return average_anti_diag(new_trajectory_matrix)
