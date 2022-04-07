# SSA
使用numpy实现的奇异谱分析的去噪方法
直接调用该方法，即可实现序列数据的去噪
def ssa(seq, window_size=None, top_k_component=None):
    """
    :author: 高世平 2021-07-16
    :func: 将输入的任意长度向量使用ssa方法去除噪声
    :param seq: np.ndarray 任意长度的一维向量
    :param window_size: 结果的稳定性，该值越大，计算量越大，但结果更稳定
    :param top_k_component: 去噪程度，该值越小，去噪程度越高
    :return: 经过ssa去噪后的结果
    """
