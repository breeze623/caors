import numpy as np

# 优化方向:1.尝试使用矩阵运算代替循环 2.尝试先进行预处理,排除标签向量差距明显的其他用户

def cal_similarity(target_user_ID, target_user_label, other_user_ID, other_user_label):
    '''
    计算目标用户与其他用户标签的相似度(距离)
    :param target_user_ID:目标用户ID
    :param target_user_label:目标用户标签向量(列向量)
    :param other_user_ID:其他用户ID(list)
    :param other_user_label:其他用户标签向量矩阵(一列为一个用户的标签向量)
    :return:最佳匹配用户的ID
    '''
    most_similar_index = 0
    for i in range(1, len(other_user_label[0])):
        most_similar = np.linalg.norm(target_user_label - other_user_label[:, most_similar_index])
        temp = np.linalg.norm(target_user_label - other_user_label[:, i])
        if temp < most_similar:
            most_similar_index = i

    return other_user_ID[i]
