import numpy as np
import pandas as pd


def LFM_grad_desc(R, P, Q, max_iter, lamda, K=2, alpha=0.0001):
    '''
    针对整个矩阵进行分解
    :param R:评分矩阵
    :param P:用户向量矩阵
    :param Q:物品向量矩阵
    :param max_iter:最大迭代次数
    :param lamda: 正则化系数
    :param K: 隐向量长度
    :param alpha: 学习率
    :return:返回训练的到的P、Q矩阵以及训练完成的损失值
    '''
    # 基本维度参数定义
    M = len(R)
    N = len(R[0])


    # 开始迭代
    for step in range(max_iter):
        # 对所有用户u、物品i做遍历,对应的特征向量Pu、Qi梯度下降
        for u in range(M):
            for i in range(N):
               # 对于每个大于0的评分,求出评分误差
                if R[u][i] > 0:
                    eui = np.dot(P[u, :], Q[:, i]) - R[u][i]

                    # 代入公式,梯度下降更新Pu,Qi
                    for k in range(K):
                        P[u][k] -= alpha * (2 * eui * Q[k][i] + 2 * lamda * P[u][k])
                        Q[k][i] -= alpha * (2 * eui * Q[k][i] + 2 * lamda * Q[k][i])
        # # 计算预测评分矩阵
        # predR = np.dot(P, Q)

        # 计算当前损失值
        cost = cost_function(P, Q, R, K, lamda)
        if cost < 0.0001:
            break

    return P, Q, cost

def LFM_grad_desc_row(R, row, P, Q, max_iter, lamda, K=2, alpha=0.0001):
    '''
    对输入矩阵的部分行进行分解
    :param R: 评分矩阵
    :param row: 要处理的行数
    :param P: 用户向量矩阵
    :param Q: 物品向量矩阵
    :param max_iter: 最大迭代次数
    :param lamda: 正则化系数
    :param K: 隐向量长度
    :param alpha: 学习率
    :return: 返回训练的到的P、Q矩阵以及训练完成的损失值
    '''
    # 基本维度参数定义
    N = len(R[0])

    # 拼接矩阵的初始化
    temp_p = np.random.rand(len(row), K)

    # 开始迭代
    for step in range(max_iter):
        # 对新增用户u、物品i做遍历,对应的特征向量temp_pu、Qi梯度下降
        for u in row:
            for i in range(N):
                # 对于每个大于0的评分,求出评分误差
                if R[u][i] > 0:
                    eui = np.dot(temp_p[u, :], Q[:, i]) - R[u][i]

                    # 代入公式,梯度下降更新temp_pu,Qi
                    for k in range(K):
                        temp_p[u][k] -= alpha * (2 * eui * Q[k][i] + 2 * lamda * temp_p[u][k])
                        Q[k][i] -= alpha * (2 * eui * Q[k][i] + 2 * lamda * Q[k][i])

        # 计算当前损失值
        cost = cost_function(temp_p, Q, R, K, lamda)
        if cost < 0.0001*len(row)/len(R):
            break

    np.vstack((P, temp_p))

    return P, Q, cost

def LFM_grad_desc_column(R, column, P, Q, max_iter, lamda, K=2, alpha=0.0001):
    '''
    对输入矩阵的部分行进行分解
    :param R: 评分矩阵
    :param column: 要处理的列数
    :param P: 用户向量矩阵
    :param Q: 物品向量矩阵
    :param max_iter: 最大迭代次数
    :param lamda: 正则化系数
    :param K: 隐向量长度
    :param alpha: 学习率
    :return: 返回训练的到的P、Q矩阵以及训练完成的损失值
    '''
    # 基本维度参数定义
    M = len(R)

    # 拼接矩阵的初始化
    temp_q = np.random.rand(K, len(column))

    # 开始迭代
    for step in range(max_iter):
        # 对所有用户u、新增物品i做遍历,对应的特征向量Pu、temp_qi梯度下降
        for u in range(M):
            for i in column:
                # 对于每个大于0的评分,求出评分误差
                if R[u][i] > 0:
                    eui = np.dot(P[u, :], temp_q[:, i]) - R[u][i]

                    # 代入公式,梯度下降更新Pu,Qi
                    for k in range(K):
                        P[u][k] -= alpha * (2 * eui * Q[k][i] + 2 * lamda * P[u][k])
                        temp_q[k][i] -= alpha * (2 * eui * Q[k][i] + 2 * lamda * temp_q[k][i])

        # 计算当前损失值
        cost = cost_function(P, temp_q, R, K, lamda)
        if cost < 0.0001*len(column)/len(R[0]):
            break

    np.hstack((Q, temp_q))

    return P, Q, cost

def cost_function(P, Q, R, K, lamda):
    cost = 0
    for i in range(len(R)):
        for j in range(len(R[0])):
            if R[i][j] > 0:
                cost += (np.dot(P[i, :], Q[:, j]) - R[i][j])**2
                # 加上正则项
                for k in range(K):
                    cost += lamda * (P[i][k]**2 + Q[k][j]**2)
    return cost




