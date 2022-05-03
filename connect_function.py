import pymysql as pm



# host = '120.25.125.57',         # 数据库地址
# port = 3306,                  # 端口号
# user = 'root',                  # 用户名
# password = '1346798520lhy',     # 密码
# database = 'zhu'
#



def connection_mysql(host, port, user, password, database):
    '''
    连接云端mysql数据库
    :param host: 数据库IP地址(String)
    :param port: 端口号(int)
    :param user: 用户名(String)
    :param password: 密码(String)
    :param database: 数据库名称(String)
    :return: 数据库、游标(使用完记得调用close函数关闭连接)
    '''

    # 数据库连接参数
    db = pm.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database
    )

    return db

def select_data(cursor, attribute_name, table_name):
    '''
    从数据库的表中获取属性值
    :param cursor:游标
    :param attribute_name:属性名称(list,其中元素为String)
    :param table_name: 表名(String)
    :return: 数据迭代器
    '''
    temp = ''
    for i in range(len(attribute_name)-1):
        temp += attribute_name[i]
        temp += ', '
    temp += attribute_name[-1]
    sql = 'SELECT {} FROM {}'.format(temp, table_name)
    cursor.execute(sql)
    records = cursor.fetchall()
    return records


def update_data(db, cursor, attribute_name, new_value, table_name, match_conditions):
    '''
    对数据库中的数据进行更新
    :param db: 数据库
    :param cursor: 游标
    :param attribute_name: 属性名称(list,其中元素为String)
    :param new_value: 属性更新值(list,其中元素为该属性值域取值)
    :param table_name: 表名(String)
    :param match_conditions: 匹配条件(String)
    :return: none
    '''

    # 若干个属性值需要修改，匹配一条数据
    temp = ''
    for i in range(len(attribute_name)-1):
        temp += attribute_name[i]+' = '+new_value[i]
        temp += ', '
    temp += attribute_name[-1]+' = '+new_value[-1]
    sql = 'UPDATE {} SET {} WHERE {}'.format(table_name, temp, match_conditions)
    cursor.execute(sql)
    db.commit()




def close(db, cursor):
    '''
    关闭连接
    :param db:数据库
    :param cursor: 游标
    :return:none
    '''
    cursor.close()
    db.close()

