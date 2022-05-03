import numpy as np

def Pretreatment(target_label, other_label):
    '''
    根据标签向量进行预处理，排除差距明显的其他label
    :param target_label: 目标标签向量(列向量)
    :param other_label: 候选标签矩阵(一列为一个标签向量)
    :return: 匹配度前200的索引
    '''
    # 前200可能会把最佳的排除，因为这里是矩阵运算，不是欧式距离，需要根据数据量及标签进行动态调整

    temp = np.dot(target_label,other_label)                                     # 矩阵乘法
    index = np.argpartition(temp, [i for i in range(200)])
    return index


def get_most_similar_userID(target_user_ID, target_user_label, other_user_ID, other_user_label):
    '''
    计算目标用户与其他用户标签的相似度(欧式距离)
    :param target_user_ID:目标用户ID
    :param target_user_label:目标用户标签向量(列向量)
    :param other_user_ID:其他用户ID(list)
    :param other_user_label:其他用户标签向量矩阵(一列为一个用户的标签向量)
    :return:最佳匹配50个用户的ID
    '''
    index = Pretreatment(target_user_label, other_user_label)
    temp = np.linalg.norm(target_user_label - other_user_label[:, index], axis=0)   # 按列计算欧式距离
    most_similar_index = np.argpartition(temp, [i for i in range(50)])              # 选择距离最小的50个用户

    return other_user_ID[most_similar_index]


def get_most_similar_articleID(target_article_ID, target_article_label, other_article_ID, other_article_label):
    '''
    计算目标文章与其他文章标签的相似度(欧式距离)
    :param target_article_ID:目标文章的ID
    :param target_article_label:目标文章的标签向量(列向量)
    :param other_article_ID:其他文章ID(list)
    :param other_article_label:其他文章标签向量矩阵(一列为一篇文章的标签向量)
    :return:最佳匹配的50篇文章的ID
    '''
    index = Pretreatment(target_article_label, other_article_label)
    temp = np.linalg.norm(target_article_label - other_article_label[:, index], axis=0)     # 按列计算欧式距离
    most_similar_index = np.argpartition(temp, [i for i in range(50)])                      # 选择距离最小的50篇文章

    return other_article_ID[most_similar_index]
