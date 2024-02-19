# -*- coding:utf-8 -*-
"""
 作者: QL
 日期: 2022年05月05日
 内容：爬取国外体育网站——”GOAL“中过去400天中超联赛（21赛季21年4月下旬开赛）每场比赛技术统计数据的json文件
"""

import datetime
import logging
import re
import os
import time
import urllib.error
import urllib.request
from functools import wraps
from typing import Callable, List, NoReturn

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

# 日志定义
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(thread)d %(levelname)s %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局静态变量
# LEAGUE = {'csl': '82jkgccg7phfjpd0mltdl3pat',  # 中超
#           'premier-league': '2kwbbcootiqqgmrzs6o5inle5',  # 英超
#           'primera-division': '34pl8szyvrbwcmfkuocjm3r6t',  # 西甲
#           'serie-a': '1r097lpxe0xn03ihb7wi98kao',  # 意甲
#           'bundesliga': '6by3h89i2eykc341oz7lv1ddd',  # 德甲
#           'ligue-1': 'dm5ka0os1e3dxcp3vh05kmp33'  # 法甲
#           }

LEAGUE = {
          'primera-division': '34pl8szyvrbwcmfkuocjm3r6t',  # 西甲
          'serie-a': '1r097lpxe0xn03ihb7wi98kao',  # 意甲
          'bundesliga': '6by3h89i2eykc341oz7lv1ddd',  # 德甲
          'ligue-1': 'dm5ka0os1e3dxcp3vh05kmp33'  # 法甲
          }


PERIOD = 400

def retry(num: int):
    """
    重试装饰器，若被装饰函数执行失败，则重试num次
    :param num: 重试次数
    """
    def retry_decorator(func: Callable):
        @wraps(func)
        def retry_wrapper(*args, **kwargs):
            for i in range(num + 1):
                if i > 0:
                    logger.info('Retry: %d/%d' % (i, num))
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error('Error: %s' % e, exc_info=True)
                    time.sleep(np.random.randint(3))  # 随机sleep时间

        return retry_wrapper

    return retry_decorator


def main():
    for key, val in LEAGUE.items():
        do_one(key, val, PERIOD)


def do_one(league: str, feature_path: str, period: int) -> NoReturn:
    """
    得到一个联赛的数据
    :param league: 联赛名
    :param feature_path: 各联赛fixtures-results页面各自的路径
    :param period: 需要爬取的时间长度
    """
    # baseurl = "https://www.goal.com/en/csl/fixtures-results/2022-05-08/82jkgccg7phfjpd0mltdl3pat"  # 2022-5-8中超赛事网页
    container = []  # 存放含有数据的json文件的目标链接以及比赛队伍名字
    daterange = pd.date_range(end=datetime.datetime.now(), periods=period)  # 生成时间序列
    datelst = daterange.strftime('%Y-%m-%d')  # 转换成字符串
    logger.debug("datelist: {}".format(datelst))

    for date in datelst:  # 遍历时间序列,得到每日赛事详情url的container
        # sub_url = baseurl[:44] + "/" + date + "/82jkgccg7phfjpd0mltdl3pat"  # 生成需要遍历的url（按照日期遍历）
        sub_url = r'https://www.goal.com/en/' + league + r'/fixtures-results/' + date + r'/' + feature_path
        logger.debug("sub_url: {}".format(sub_url))
        html = askURL(sub_url)  # 获得html
        soup = BeautifulSoup(html, "html.parser")
        lst = soup.find_all('a', class_="match-main-data-link")  # 找到网页上的所有赛事详情超链接对应的标签
        logger.debug("lst: {}".format(lst))
        link_lst = find_url(lst)  # 提取每日不同赛事详情的url
        for href in link_lst:  # 遍历上述url
            sub_sub_url = "https://www.goal.com" + href
            logger.debug("sub_sub_url: {}".format(sub_sub_url))
            # match = re.search("(?<=/en/match/).+?(?=/)", href).group()  # 正则表达搜索比赛名称(该html中西语、法语乱码)
            # match = match + '/' + date
            sub_html = askURL(sub_sub_url)  # 得到比赛详情页面
            # sub_soup = BeautifulSoup(sub_html, "html.parser")
            # match_name_lst = sub_soup.find_all('div', class_="widget-match-header__name")
            link = find_json(sub_html)  # 找到json文件的链接并返回
            match_dic = find_match_name(sub_html, link, date)
            if link:
                container.append(match_dic)  # 把json文件链接放到container中

    if os.path.exists(league):  # 判断文件夹是否存在
        logger.debug('Exist {} folder already'.format(league))
    else:
        os.mkdir(league)

    if len(container) > 0:
        for i in range(len(container)):  # 遍历生成从0到len(container)为文件名的json文件
            for key, val in container[i].items():
                val = val.replace('\\', '')  # 对链接字符串进行修正处理
                save_json(league, key, askURL(val))  # 发起请求，并保存json文件信息
    else:
        logger.warning('{} has no data link'.format(league))


def find_url(lst: list) -> List[str]:
    """
    提取每日不同赛事详情的url
    :param lst: 存放每日不同赛事详情的url的列表
    """
    link_lst = []
    for i in lst:
        link = re.search("(?<=href=\").+?(?=\")", str(i)).group()  # 提取赛事详情url
        logger.debug("The gotten Hypertext Reference is: {}".format(link))
        link_lst.append(link)
    return link_lst


def find_match_name(html: str, json_link: str, date: str) -> dict:
    """
    找到比赛球队名字
    :param html: 赛事详情页面
    :param json_link: 该赛事对应的json文件链接
    :param date: 比赛日
    """
    match_dic = {}  # 字典，存放比赛名、日期和统计数据link
    sub_soup = BeautifulSoup(html, "html.parser")
    match_name_lst = sub_soup.find_all('div', class_="widget-match-header__name")
    logger.debug('match_name_lst: {}'.format(match_name_lst))
    team_name_lst = []  # 暂时存放两队名字信息
    for i in match_name_lst:
        team_name = re.search("(?<=itemprop=\"name\">).+?(?=<)", str(i)).group()  # 查找比赛名字
        team_name_lst.append(team_name)
    match_name_str = team_name_lst[0] + '-vs-' + team_name_lst[1] + '_' + date  # 重新定义比赛名字字符串
    match_dic[match_name_str] = json_link
    return match_dic



def find_json(html: str) -> str:
    """
    找到html中存放数据的json链接，并下载下来
    :param html: 请求返回的HTTPResponse解码后的字符串
    """
    link = ''
    soup = BeautifulSoup(html, "html.parser")
    result = soup.find_all('div', class_='widget-match-key-events')  # 找到json文件的标签
    logger.debug("result: {}".format(result))
    if result:
        link = re.search("(?<=endpointUrl\":\").+?(?=\")", str(result)).group()  # 正则表达找到json文件的链接
    return link


def save_json(league: str, file_name: str, file_content: str) -> NoReturn:
    """
    保存json文件
    :param file_name: 文件名
    :param file_content: 文件内容
    """
    with open('./' + league + '/' + file_name.replace('/', '_') + '.json', 'wb') as f:
        f.write(file_content.encode())  # 对str编码，写入bytes类型数据
    logger.debug('save {}.json successfully in folder {}'.format(file_name, league))


@retry(3)
def askURL(url: str) -> str:
    """
    发起请求，返回html
    """
    headers = {  # 模拟头部信息
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36"
    }
    request = urllib.request.Request(url, headers=headers)
    html = ""
    response = urllib.request.urlopen(request)
    html = response.read().decode("utf-8")  # str
    response.close()  # 关闭response
    return html


if __name__ == "__main__":
    main()
    logger.debug("Done!")
