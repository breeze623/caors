import numpy as np
import LFM_function as fun

# 评分矩阵
R = np.array([[4, 0, 2, 0, 1],
              [0, 2, 3, 0, 0],
              [1, 0, 2, 4, 0],
              [5, 0, 0, 3, 1],
              [0, 0, 1, 5, 1],
              [0, 3, 2, 4, 1]])

# 给定超参数
K = 5
max_iter = 5000
alpha = 0.0002
lamda = 0.004


# 测试
P, Q, cost = fun.LFM_grad_desc(R, max_iter, lamda, K, alpha)
predR = np.dot(P, Q)
