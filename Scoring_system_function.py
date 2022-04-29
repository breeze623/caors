import numpy as np

def heat_scoring(matrix, Class=0):
    '''
    计算文章热度
    :param matrix:文章/向量组成的矩阵(每一列为一篇文章的属性)
        默认属性排序: ID, 浏览次数, 文本长度, 点赞数量, 关注数量, 评论数量, 热度值
    :param Class: 矩阵包含类型(问题0 or 文章1)
    :return: 得出文章热度
    '''
    weight_num_of_views = 0.0001                    # 浏览次数的权重
    weight_length_of_content = 0.0001               # 文本长度的权重
    weight_num_of_likes = 0.0001                    # 点赞数量的权重
    weight_num_of_subscriptions = 0.0001            # 关注数量的权重
    weight_num_of_commons = 0.0001                  # 评论数量的权重

    list = [weight_num_of_views, weight_length_of_content, weight_num_of_likes, weight_num_of_subscriptions, weight_num_of_commons]
    W = np.array(list)                              # 权重向量


    # 对文章的文本长度进行处理
    if (Class==1):
        matrix[2, :] = np.log(matrix[2, :])
        heat_score = np.dot(W, matrix[1:-2, :])

    else:
        W[1] = 0                                    # 问题不考虑文本长度的影响
        heat_score = np.dot(W, matrix[1:-2, :])

    return heat_score

