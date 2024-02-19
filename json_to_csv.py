# -*- coding:utf-8 -*-
"""
 作者: QL
 日期: 2022年05月07日
 内容: 将爬取到的json文件中的数据提取出来并保存到csv文件中
"""
import os
import json
import re

import pandas as pd
import logging
from typing import NoReturn

# 日志定义
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(thread)d %(levelname)s %(module)s - %(message)s')
logger = logging.getLogger(__name__)

LEAGUE = ['csl', 'premier-league', 'primera-division', 'serie-a', 'bundesliga', 'ligue-1']  # 联赛


def main():
    example_path = r'.\csl\Beijing Guoan-vs-Changchun Yatai_2021-08-12.json'  # 一个用于新建初始dataframe的json文件路径
    df_empty = get_base_dataframe(example_path)  # 得到初始dataframe

    # team_id_name_dict = get_team_id_name_dict()

    for league in LEAGUE:
        file_path = league  # json文件目录
        # n = count_json(file_path)  # json文件个数
        df_with_data = json_to_dataframe(file_path, df_empty)  # 把该目录下的json文件中的数据添加到dataframe中
        target_path = league  # 存放csv文件的目标目录
        dataframe_to_csv(target_path, df_with_data, league)


# def get_team_id_name_dict() -> dict:
#     """
#     得到球队id和名字的字典
#     """
#     team_id_name_dict = {}
#     for league in LEAGUE:
#         with open(r'.\team_data' + '\\' + league + '.json', 'rb') as f:
#             result = json.load(f)
#             for i in result['standings']:
#                 team_id_name_dict[i['teamId']] = i['teamName']
#     return team_id_name_dict


# def count_json(path: str) -> int:
#     """
#     统计目录下json文件数
#     :param path: json文件存在的目录
#     """
#     count = 0
#     if os.path.exists(path):  # 判断目录是否存在
#         files = os.listdir(path)
#         for i in files:
#             if os.path.splitext(i)[1] == '.json':  # 分离文件名和后缀
#                 count += 1
#     else:
#         logger.warning('No such path')
#     return count


def get_base_dataframe(path: str) -> pd.DataFrame:
    with open(path) as f:
        result = json.load(f)
    lst = []  # 新建空列表，存放属性
    for key, val in result['stats'].items():  # 遍历一个数据，找出属性信息
        lst.append(key)  # 把属性信息存入列表
    lst.append('Home/Away')  # 增加主客场属性
    lst.append('Team')  # 增加球队名属性
    lst.append('Win')  # 增加胜负属性（胜1，负0，平2）
    logger.debug("lst{}".format(lst))
    return pd.DataFrame(columns=lst)  # 返回含有上述属性的dataframe


def json_to_dataframe(file_path: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    把json文件数据转换成dataframe
    :param file_path: json文件目录
    :param df:
    """
    filenames = os.listdir(file_path)
    for i in filenames:
        if os.path.splitext(i)[1] == '.json':
            with open(file_path + '\\' + i) as f:
                result = json.load(f)
            # 根据比赛名匹配主客场球队名
            home_name = re.search(".+?(?=-vs-)", i).group()
            away_name = re.search("(?<=-vs-).+?(?=_)", i).group()

            # 统计主客场进球
            lst = result['keyEvents']['all']
            home = 0  # 主场进球计数器
            away = 0  # 客场进球计数器
            for i in range(len(lst)):  # 循环事件记录
                if 'goal' in lst[i]['type'] and 'var' not in lst[i]['type']:  # 找到进球记录
                    logger.debug('Goal record:\n{}'.format(lst[i]))
                    if lst[i]['teamType'] == 'home':  # 判断进球球队
                        home += 1  # 增加计数
                    elif lst[i]['teamType'] == 'away':
                        away += 1
                    else:
                        pass

            # 将主客场信息分别存放到主客场数据字典中，然后添加到dataframe中
            dic_home = dict()  # 新建主场数据字典
            dic_away = dict()  # 新建客场数据字典
            logger.debug('stats:\n{}'.format(result['stats']))
            for key, val in result['stats'].items():  # 循环stats数据记录，分别将主客场信息存放到相应的字典中
                for sub_key, sub_val in result['stats'][key].items():
                    if sub_key == 'home':
                        dic_home[key] = sub_val
                    elif sub_key == 'away':
                        dic_away[key] = sub_val
                    else:
                        pass
            dic_home['Home/Away'] = 'home'
            dic_away['Home/Away'] = 'away'
            dic_home['Team'] = home_name
            dic_away['Team'] = away_name
            if home > away:
                dic_home['Win'] = 1
                dic_away['Win'] = 0
            elif home < away:
                dic_home['Win'] = 0
                dic_away['Win'] = 1
            elif home == away:
                dic_home['Win'] = 2
                dic_away['Win'] = 2
            df = df.append(dic_home, ignore_index=True)
            df = df.append(dic_away, ignore_index=True)
    return df


def dataframe_to_csv(path: str, df: pd.DataFrame, league: str) -> NoReturn:
    """
    把dataframe保存成csv格式文件
    :param path: 保存csv文件路径
    :param df: 带有数据的dataframe
    :param league: 以联赛名作为csv文件名
    """
    if os.path.exists(path):
        df.to_csv(path + '\\' + league + '.csv')
    else:
        logger.warning('No such path!')


if __name__ == "__main__":
    main()
    logger.debug("Done!")
