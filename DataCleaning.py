# -*- coding:utf-8 -*-
"""
 作者: QL
 日期: 2022年04月30日
 内容: 利用继承和多态封装一个数据预处理的模块
 PS: 虽然第九章没学，但我还是试着去完善输入输出数据和方法的约定，如有不当希望老师能够指出，谢谢老师！
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import logging
from typing import Optional, NoReturn, TypeVar, NewType, List

# 日志定义
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(thread)d %(levelname)s %(module)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 类型定义
DataFrame = NewType('DataFrame', pd.DataFrame)
Series = NewType('Series', pd.Series)
Bound = TypeVar('Bound', int, float)
Columns = TypeVar('Columns', str, List[str])
Values = TypeVar('Values', int, float, str)


class Outlier:
    """
    异常值处理
    """

    def __init__(self, df: DataFrame):
        self.df = df

    def experience_drop(self,
                        col: str,
                        lower_bound: Optional[Bound] = None,
                        upper_bound: Optional[Bound] = None) -> NoReturn:
        """
        根据经验寻找异常值并删除
        :param col: 存在异常值的列
        :param lower_bound: 下界
        :param upper_bound: 上界
        """
        if lower_bound is None and upper_bound is None:  # 判断是否有上下界
            logging.warning('please passing lower_bound or upper_bound')
        else:
            try:
                if lower_bound is None and upper_bound is not None:
                    self.df.drop(self.df[self.df[col] > upper_bound].index, inplace=True)  # 删除异常值
                elif lower_bound is not None and upper_bound is None:
                    self.df.drop(self.df[self.df[col] < lower_bound].index, inplace=True)
                elif lower_bound is not None and upper_bound is not None:
                    self.df.drop(self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].index,
                                 inplace=True)
            except BaseException as e:
                logger.error(e)
            else:
                logger.info("Successfully drop outliers for {} with given bounds".format(col))

    def experience_replace(self,
                           col: str,
                           method: str,
                           lower_bound: Optional[Bound] = None,
                           upper_bound: Optional[Bound] = None) -> NoReturn:
        """
        根据经验寻找异常值并替换
        :param col: 存在异常值的列
        :param method: 异常值替换方法
        :param lower_bound: 下界
        :param upper_bound: 上界
        """

        def choose_method(method1: str, case1: str, df11: DataFrame) -> NoReturn:
            """
            选择方法替换异常值
            :param method1: 方法
            :param case1: 对应的三种情况
            :param df11: 非异常值对应的DataFrame
            PS:这一部分写的有点复杂，主要是因为dataframe修改的时候不能通过视图对原数据进行修改
            """
            try:
                if method1 == 'mean':  # 判断方法
                    a1 = df11.loc[:, col].mean()
                    if case1 == 'case1':  # 判断情况
                        self.df.loc[(self.df[col] > upper_bound), col] = a1  # 替换数据
                    elif case1 == 'case2':
                        self.df.loc[(self.df[col] < lower_bound), col] = a1
                    elif case1 == 'case3':
                        self.df.loc[((self.df[col] < lower_bound) | (self.df[col] > upper_bound)), col] = a1
                elif method1 == 'median':
                    a1 = df11.loc[:, col].median()
                    if case1 == 'case1':
                        self.df.loc[(self.df[col] > upper_bound), col] = a1
                    elif case1 == 'case2':
                        self.df.loc[(self.df[col] < lower_bound), col] = a1
                    elif case1 == 'case3':
                        self.df.loc[((self.df[col] < lower_bound) | (self.df[col] > upper_bound)), col] = a1
                elif method1 == 'mode':
                    a1 = df11.loc[:, col].mode()
                    if case1 == 'case1':
                        self.df.loc[(self.df[col] > upper_bound), col] = a1
                    elif case1 == 'case2':
                        self.df.loc[(self.df[col] < lower_bound), col] = a1
                    elif case1 == 'case3':
                        self.df.loc[((self.df[col] < lower_bound) | (self.df[col] > upper_bound)), col] = a1
                elif method1 == 'bfill':
                    if case1 == 'case1':
                        self.df.loc[(self.df[col] > upper_bound), col] = np.nan
                    elif case1 == 'case2':
                        self.df.loc[(self.df[col] < lower_bound), col] = np.nan
                    elif case1 == 'case3':
                        self.df.loc[((self.df[col] < lower_bound) | (self.df[col] > upper_bound)), col] = np.nan
                    self.df.fillna(method='bfill', inplace=True)
                elif method1 == 'ffill':
                    if case1 == 'case1':
                        self.df.loc[(self.df[col] > upper_bound), col] = np.nan
                    elif case1 == 'case2':
                        self.df.loc[(self.df[col] < lower_bound), col] = np.nan
                    elif case1 == 'case3':
                        self.df.loc[((self.df[col] < lower_bound) | (self.df[col] > upper_bound)), col] = np.nan
                    self.df.fillna(method='ffill', inplace=True)
                else:
                    raise ValueError('No such method!')
            except BaseException as e1:
                logger.error(e1)
            else:
                logger.info('Successfully using {} to replace the outliers of {}'.format(method1, col))

        if lower_bound is None and upper_bound is None:  # 判断是否有上下界
            logging.warning('please passing lower_bound or upper_bound')
        else:
            try:
                if lower_bound is None and upper_bound is not None:
                    case = 'case1'
                    # df0 = self.df[self.df[col] > upper_bound]  # 异常值 PS:本来是想通过df0这个视图对数据进行替换操作的，但是不行
                    df1 = self.df[self.df[col] <= upper_bound]  # 非异常值
                    choose_method(method, case, df1)  # 调用函数替换异常值
                elif lower_bound is not None and upper_bound is None:
                    case = 'case2'
                    # df0 = self.df[self.df[col] < lower_bound]  # 异常值
                    df1 = self.df[self.df[col] >= lower_bound]  # 非异常值
                    choose_method(method, case, df1)
                elif lower_bound is not None and upper_bound is not None:
                    case = 'case3'
                    # df0 = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]  # 异常值
                    df1 = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]  # 非异常值
                    choose_method(method, case, df1)
            except BaseException as e:
                logger.error(e)  # 如果choose_method里出错了，这里应该会重复写一次error

    def experience_get(self,
                       col: str,
                       lower_bound: Optional[Bound] = None,
                       upper_bound: Optional[Bound] = None) -> DataFrame:
        """
        根据经验得到某列为异常值的DataFrame
        :param col: 存在异常值的列
        :param lower_bound: 下界
        :param upper_bound: 上界
        """
        if lower_bound is None and upper_bound is None:  # 判断是否有上下界
            logging.warning('please passing lower_bound or upper_bound')
        elif lower_bound is not None or upper_bound is not None:
            try:
                if lower_bound is None and upper_bound is not None:
                    return self.df[self.df[col] > upper_bound]  # 返回异常值
                elif lower_bound is not None and upper_bound is None:
                    return self.df[self.df[col] < lower_bound]
                elif lower_bound is not None and upper_bound is not None:
                    return self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            except BaseException as e:
                logger.error(e)
            finally:
                logger.info('Get it. lower_bound is {} and upper_bound is {}'.format(lower_bound, upper_bound))

    def is_normal(self, col: Columns) -> NoReturn:
        """
        用K-S检验判断数据是否为normal distribution
        :param col: 需要判断的列
        """
        if type(col) == str:
            u = self.df[col].mean()
            std = self.df[col].std()
            logger.info("mean:%.2f, standard deviation:%.2f" % (u, std))
            statistic, pvalue = stats.kstest(self.df[col], 'norm',
                                             (u, std))  # 计算统计量和p值，前者越趋近1越服从正态分布，p值大于0.05则显著服从正态分布正态分布
            logger.info('statistic is {}, pvalue is {}'.format(statistic, pvalue))
            if pvalue > 0.05:  # p值大于0.05，显著，为正态分布
                logger.info('The data of {} follow normal distribution'.format(col))
            else:
                logger.info('The data of {} do not follow normal distribution'.format(col))
        elif type(col) == list:
            df0 = pd.DataFrame(np.zeros((5, len(col))),
                               columns=col,
                               index=['mean', 'std', 'statistic', 'pvalue', 'normal'])  # 建立输出DataFrame
            for i in col:
                u = self.df[i].mean()
                std = self.df[i].std()
                statistic, pvalue = stats.kstest(self.df[i], 'norm', (u, std))
                if pvalue > 0.05:
                    judgement = 'Yes'
                else:
                    judgement = 'No'
                lst = [u, std, statistic, pvalue, judgement]
                df0.loc[:, i] = lst  # 对输出DataFrame填入计算数据
            logger.info('\n{}'.format(df0))
        else:
            logger.warning('Column type is wrong!')

    def three_sigma_drop(self, col: Columns) -> NoReturn:
        """
        利用3σ准则删除近似正态分布数据的异常值
        :param col: 需要删除异常值的列
        """
        if type(col) == str:
            u = self.df[col].mean()
            std = self.df[col].std()
            self.df.drop(self.df[(self.df[col] < (u - 3 * std)) | (self.df[col] > (u + 3 * std))].index,
                         inplace=True)  # 删除异常值
            logger.info("Successfully drop outliers for {} with '3σ' method".format(col))
        elif type(col) == list:
            for i in col:
                u = self.df[i].mean()
                std = self.df[i].std()
                if type(i) == str:
                    self.df.drop(self.df[(self.df[i] < (u - 3 * std)) | (self.df[i] > (u + 3 * std))].index,
                                 inplace=True)
                    logger.info("Successfully drop outliers for {} with '3σ' method".format(i))
                else:
                    logger.warning('Column {} type error'.format(i))
        else:
            logger.warning('Please passing correct arguments')

    def three_sigma_replace(self, **kwargs) -> NoReturn:
        """
        利用3σ准则替换近似正态分布数据的异常值
        :param kwargs: 需要替换异常值的列以及替换方法
        """
        for key, val in kwargs.items():
            u = self.df[key].mean()
            std = self.df[key].std()
            df0 = self.df[(self.df[key] >= (u - 3 * std)) & (self.df[key] <= (u + 3 * std))]  # 非异常值
            # df1 = self.df[(self.df[key] < (u - 3 * std)) | (self.df[key] > (u + 3 * std))]  # 异常值
            try:
                if val == 'mean':  # 判断方法
                    a = df0.loc[:, key].mean()
                    self.df.loc[((self.df[key] < (u - 3 * std)) | (self.df[key] > (u + 3 * std))), key] = a  # 替换异常值
                elif val == 'median':
                    a = df0.loc[:, key].median()
                    self.df.loc[((self.df[key] < (u - 3 * std)) | (self.df[key] > (u + 3 * std))), key] = a
                elif val == 'mode':
                    a = df0.loc[:, key].mode()
                    self.df.loc[((self.df[key] < (u - 3 * std)) | (self.df[key] > (u + 3 * std))), key] = a
                elif val == 'bfill':
                    self.df.loc[((self.df[key] < (u - 3 * std)) | (self.df[key] > (u + 3 * std))), key] = np.nan
                    self.df.fillna(method='bfill', inplace=True)
                elif val == 'ffill':
                    self.df.loc[((self.df[key] < (u - 3 * std)) | (self.df[key] > (u + 3 * std))), key] = np.nan
                    self.df.fillna(method='ffill', inplace=True)
                else:
                    raise ValueError('No such method!')
            except BaseException as e:
                logger.error(e)
            else:
                logger.info("Successfully using {} to replace the outliers of {}  with '3σ' method".format(val, key))

    def three_sigma_get(self, col: str) -> DataFrame:
        """
        按照3σ法则得到某列为异常值的DataFrame
        :param col: 要得到异常值的列
        """
        try:
            u = self.df[col].mean()
            std = self.df[col].std()
            logger.info('Get it. u-3σ is {} and u+3σ is {}'.format(u - 3 * std, u + 3 * std))
            return self.df[(self.df[col] < (u - 3 * std)) | (self.df[col] > (u + 3 * std))]  # 返回异常值
        except BaseException as e:
            logger.error(e)

    def boxplot(self, col: Columns, onefig: bool = True, figsize: Optional[tuple] = None) -> NoReturn:
        """
        画出数据的箱型图
        :param col: 需要作图的列
        :param onefig: 是否需要将箱型图画在一个图里
        :param figsize: figure的大小
        """
        if type(col) == str:
            plt.figure(figsize=figsize)
            plt.boxplot(self.df[col], labels=[col])
            plt.show()
        if type(col) == list:
            if onefig:
                plt.figure(figsize=figsize)
                lst = []  # 新建存放列数据的list
                for i in col:
                    lst.append(self.df[i])
                plt.boxplot(lst, labels=col)  # 画图
                plt.show()
            else:
                if len(col) < 5:  # 判断列的长度
                    fig, axes = plt.subplots(1, len(col), figsize=figsize)
                    for i in range(len(col)):
                        axes[i].boxplot(self.df[col[i]], labels=[col[i]])
                    plt.show()
                else:
                    fig, axes = plt.subplots(len(col) // 5 + 1, 5, figsize=figsize)  # 确定axes的个数
                    for i in range(len(col)):
                        axes[i // 5, i % 5].boxplot(self.df[col[i]], labels=[col[i]])  # 确定个箱型图画在哪个axes里
                    plt.show()

    def boxplot_drop(self, col: Columns) -> NoReturn:
        """
        利用箱线图删除数据的异常值
        :param col: 需要删除异常值的列
        """
        if type(col) == str:
            q1 = np.percentile(self.df[col], 25)  # 四分之一分位数
            q3 = np.percentile(self.df[col], 75)  # 四分之三分位数
            lower_bound = q1 - 1.5 * (q3 - q1)  # 异常值下界
            upper_bound = q3 + 1.5 * (q3 - q1)  # 异常值上界
            self.df.drop(self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].index,
                         inplace=True)  # 删除异常值
            logger.info("Successfully drop outliers for {} with 'boxplot' method".format(col))
        elif type(col) == list:
            for i in col:
                q1 = np.percentile(self.df[i], 25)
                q3 = np.percentile(self.df[i], 75)
                lower_bound = q1 - 1.5 * (q3 - q1)
                upper_bound = q3 + 1.5 * (q3 - q1)
                if type(i) == str:
                    self.df.drop(self.df[(self.df[i] < lower_bound) | (self.df[i] > upper_bound)].index, inplace=True)
                    logger.info("Successfully drop outliers for {} with 'boxplot' method".format(i))
                else:
                    logger.warning('Column {} type error'.format(i))
        else:
            logger.warning('Please passing correct arguments')

    def boxplot_replace(self, **kwargs) -> NoReturn:
        """
        利用箱线图法替换数据的异常值
        :param kwargs: 需要替换异常值的列以及替换方法
        """
        for key, val in kwargs.items():
            q1 = np.percentile(self.df[key], 25)
            q3 = np.percentile(self.df[key], 75)
            lower_bound = q1 - 1.5 * (q3 - q1)
            upper_bound = q3 + 1.5 * (q3 - q1)
            df0 = self.df[(self.df[key] >= lower_bound) & (self.df[key] <= upper_bound)]  # 非异常值
            # df1 = self.df[(self.df[key] < lower_bound) | (self.df[key] > upper_bound)]  # 异常值, df1是一个视图，不可级联修改
            try:
                if val == 'mean':  # 判断方法
                    a = df0.loc[:, key].mean()
                    self.df.loc[
                        ((self.df[key] < lower_bound) | (self.df[key] > upper_bound)), key] = a  # 替换异常值，用df1会Warning
                elif val == 'median':
                    a = df0.loc[:, key].median()
                    self.df.loc[((self.df[key] < lower_bound) | (self.df[key] > upper_bound)), key] = a
                elif val == 'mode':
                    a = df0.loc[:, key].mode()
                    self.df.loc[((self.df[key] < lower_bound) | (self.df[key] > upper_bound)), key] = a
                elif val == 'bfill':
                    self.df.loc[((self.df[key] < lower_bound) | (self.df[key] > upper_bound)), key] = np.nan
                    self.df.fillna(method='bfill', inplace=True)
                elif val == 'ffill':
                    self.df.loc[((self.df[key] < lower_bound) | (self.df[key] > upper_bound)), key] = np.nan
                    self.df.fillna(method='ffill', inplace=True)
                else:
                    raise ValueError('No such method!')
            except BaseException as e:
                logger.error(e)
            else:
                logger.info(
                    "Successfully using {} to replace the outliers of {}  with 'boxplot' method".format(val, key))

    def boxplot_get(self, col: str) -> DataFrame:
        """
        按照箱线图法得到某列为异常值的DataFrame
        :param col: 要得到异常值的列
        """
        try:
            q1 = np.percentile(self.df[col], 25)
            q3 = np.percentile(self.df[col], 75)
            lower_bound = q1 - 1.5 * (q3 - q1)
            upper_bound = q3 + 1.5 * (q3 - q1)
            logger.info('Get it. lower_bound is {} and upper_bound is {}'.format(lower_bound, upper_bound))
            return self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]  # 返回异常值
        except BaseException as e:
            logger.error(e)


class DataCleaning(Outlier):  # 继承
    """
    数据预处理(清洗)
    """

    def __init__(self, df: DataFrame):
        df1 = df.copy()  # 备份数据，不对原数据造成影响
        self.df = df1
        super(DataCleaning, self).__init__(df1)

    def info(self) -> NoReturn:
        """
        输出常见数据信息
        """
        if not self.df.empty:
            logger.info('\ndataframe_shape:\n{}\n{}'.format(self.df.shape, '*' * 80))
            logger.info('\ndataframe_head(5):\n{}\n{}'.format(self.df.head(5), '*' * 80))
            logger.info('\ndataframe_tail(5):\n{}\n{}'.format(self.df.tail(5), '*' * 80))
            logger.info('\ndataframe_random_10_sample:\n{}\n{}'.format(self.df.sample(10), '*' * 80))
            logger.info('\ndataframe_value_describe:\n{}\n{}'.format(self.df.describe(), '*' * 80))
            logger.info('\ndataframe_info:\n{}\n{}'.format(self.df.info(), '*' * 80))
        else:
            logging.warning('DataFrame is empty!')

    def drop_null(self,
                  axis: Optional[int] = 0,
                  how: Optional[str] = 'any') -> NoReturn:
        """
        删除缺失值
        :param axis:删除数据的方向，0代表按行删除，1代表按列删除
        :param how:删除的方式，'all'代表某列或某行全为0时才将该列或该行删除；'any'代表某列或某行中有一个字段为0就将该列或该行删除
        """
        logger.info('Before drop:\n{}\n\n'.format(self.df.isnull().sum()))  # 看看处理前的空值情况
        if how == 'any':  # 存在空值就删除
            if axis == 0:  # 按行
                self.df.dropna(axis=0, inplace=True)
            elif axis == 1:  # 按列
                self.df.dropna(axis=1, inplace=True)
            else:
                logging.warning('axis must be 0 or 1, try again!')
        elif how == 'all':  # 全为空值才删除
            if axis == 0:
                self.df.dropna(axis=0, how='all', inplace=True)
            elif axis == 1:
                self.df.dropna(axis=1, how='all', inplace=True)
            else:
                logging.warning('axis must be 0 or 1, try again!')
        else:
            logging.warning("Unknown value for 'how',try again with 'any' or 'all'!")
        logger.info('After drop:\n{}\n\n'.format(self.df.isnull().sum()))

    def fill_null_for_all(self,
                          method: str,
                          val: Values = 0) -> NoReturn:
        """
        填充全部缺失值
        :param method: 填充方法
        :param val: 指定填充值
        """
        try:
            if method == 'val':  # 判断方法
                self.df.fillna(val, inplace=True)  # 填充缺失值
            elif method == 'mean':
                self.df.fillna(self.df.mean(), inplace=True)
            elif method == 'median':
                self.df.fillna(self.df.median(), inplace=True)
            elif method == 'mode':
                self.df.fillna(self.df.mode(), inplace=True)
            elif method == 'bfill':
                self.df.fillna(method='bfill', inplace=True)
            elif method == 'ffill':
                self.df.fillna(method='ffill', inplace=True)
            else:
                raise ValueError('No such method!')
        except BaseException as e:
            logger.error(e)
        else:
            logger.info('successfully fill with {}'.format(method))

    def fill_null(self, **kwargs) -> NoReturn:
        """
        按列填充缺失值
        :param kwargs:需传入列名和对应的方法或者数字
        """
        for key, val in kwargs.items():  # 对 列名 和 方法 进行解包
            try:
                if val == 'mean':  # 判断方法
                    self.df[key].fillna(self.df[key].mean(), inplace=True)  # 填充缺失值
                elif val == 'median':
                    self.df[key].fillna(self.df[key].median(), inplace=True)
                elif val == 'mode':
                    self.df[key].fillna(self.df[key].mode(), inplace=True)
                elif val == 'bfill':
                    self.df[key].fillna(method='bfill', inplace=True)
                elif val == 'ffill':
                    self.df[key].fillna(method='ffill', inplace=True)
                elif type(val) == int or type(val) == float or type(val) == str:
                    self.df[key].fillna(val, inplace=True)
                else:
                    raise ValueError('No such method!')
            except BaseException as e:
                logger.error(e)
            else:
                logger.info('Successfully using {} to fill the null of {} '.format(val, key))

    def get(self):
        """
        得到处理后的DataFrame
        """
        logger.info('Get cleaned data')
        return self.df


class DataCleaningForSeries:  # 多态
    def __init__(self, s: Series) -> NoReturn:
        s1 = s.copy()
        self.s = s1

    def info(self):
        """
        输出常见数据信息
        """
        if not self.s.empty:
            logger.info('\nSeries_head(5):\n{}\n{}'.format(self.s.head(5), '*' * 80))
            logger.info('\nSeries_tail(5):\n{}\n{}'.format(self.s.tail(5), '*' * 80))
            logger.info('\nSeries_random_10_sample:\n{}\n{}'.format(self.s.sample(10), '*' * 80))
            logger.info('\nSeries_value_describe:\n{}\n{}'.format(self.s.describe(), '*' * 80))
        else:
            logging.warning('Series is empty!')

    # 空值处理
    def drop_null(self) -> NoReturn:
        """
        删除缺失值
        """
        logger.info('Before drop, there are {} NaN.\n'.format(self.s.isnull().sum()))  # 看看处理前的空值情况
        self.s.dropna(inplace=True)
        logger.info('After drop, there are {} NaN.\n'.format(self.s.isnull().sum()))  # 看看处理前的空值情况

    def fill_null(self,
                  method: str,
                  val: Values = 0) -> NoReturn:
        """
        填充全部缺失值
        :param method: 填充方法
        :param val: 指定填充值
        """
        try:
            if method == 'val':  # 判断方法
                self.s.fillna(val, inplace=True)  # 填充缺失值
            elif method == 'mean':
                self.s.fillna(self.s.mean(), inplace=True)
            elif method == 'median':
                self.s.fillna(self.s.median(), inplace=True)
            elif method == 'mode':
                self.s.fillna(self.s.mode(), inplace=True)
            elif method == 'bfill':
                self.s.fillna(method='bfill', inplace=True)
            elif method == 'ffill':
                self.s.fillna(method='ffill', inplace=True)
            else:
                raise ValueError('No such method!')
        except BaseException as e:
            logger.error(e)
        else:
            logger.info('successfully fill with {}'.format(method))

    # 异常值处理
    def experience_drop(self,
                        lower_bound: Optional[Bound] = None,
                        upper_bound: Optional[Bound] = None) -> NoReturn:
        """
        根据经验寻找异常值并删除
        :param lower_bound: 下界
        :param upper_bound: 上界
        """
        if lower_bound is None and upper_bound is None:  # 判断是否有上下界
            logging.warning('please passing lower_bound or upper_bound')
        else:
            try:
                if lower_bound is None and upper_bound is not None:
                    self.s.drop(self.s[self.s > upper_bound].index, inplace=True)  # 删除异常值
                elif lower_bound is not None and upper_bound is None:
                    self.s.drop(self.s[self.s < lower_bound].index, inplace=True)
                elif lower_bound is not None and upper_bound is not None:
                    self.s.drop(self.s[(self.s < lower_bound) | (self.s > upper_bound)].index, inplace=True)
            except BaseException as e:
                logger.error(e)
            else:
                logger.info("Successfully drop outliers with given bounds")

    def experience_replace(self,
                           method: str,
                           lower_bound: Optional[Bound] = None,
                           upper_bound: Optional[Bound] = None) -> NoReturn:
        """
        根据经验寻找异常值并替换
        :param method: 异常值替换方法
        :param lower_bound: 下界
        :param upper_bound: 上界
        """

        def choose_method(method1: str, case1: str, s11: Series) -> NoReturn:
            """
            选择方法替换异常值
            :param method1: 方法
            :param case1: 对应的三种情况
            :param s11: 非异常值对应的Series
            """
            try:
                if method1 == 'mean':  # 判断方法
                    a1 = s11.mean()
                    if case1 == 'case1':  # 判断情况
                        self.s[self.s > upper_bound] = a1  # 替换数据
                    elif case1 == 'case2':
                        self.s[self.s < lower_bound] = a1
                    elif case1 == 'case3':
                        self.s[(self.s < lower_bound) | (self.s > upper_bound)] = a1
                elif method1 == 'median':
                    a1 = s11.median()
                    if case1 == 'case1':  # 判断情况
                        self.s[self.s > upper_bound] = a1  # 替换数据
                    elif case1 == 'case2':
                        self.s[self.s < lower_bound] = a1
                    elif case1 == 'case3':
                        self.s[(self.s < lower_bound) | (self.s > upper_bound)] = a1
                elif method1 == 'mode':
                    a1 = s11.mode()
                    if case1 == 'case1':  # 判断情况
                        self.s[self.s > upper_bound] = a1  # 替换数据
                    elif case1 == 'case2':
                        self.s[self.s < lower_bound] = a1
                    elif case1 == 'case3':
                        self.s[(self.s < lower_bound) | (self.s > upper_bound)] = a1
                elif method1 == 'bfill':
                    if case1 == 'case1':
                        self.s[self.s > upper_bound] = np.nan
                    elif case1 == 'case2':
                        self.s[self.s < upper_bound] = np.nan
                    elif case1 == 'case3':
                        self.s[(self.s < lower_bound) | (self.s > upper_bound)] = np.nan
                    self.s.fillna(method='bfill', inplace=True)
                elif method1 == 'ffill':
                    if case1 == 'case1':
                        self.s[self.s > upper_bound] = np.nan
                    elif case1 == 'case2':
                        self.s[self.s < upper_bound] = np.nan
                    elif case1 == 'case3':
                        self.s[(self.s < lower_bound) | (self.s > upper_bound)] = np.nan
                    self.s.fillna(method='ffill', inplace=True)
                else:
                    raise ValueError('No such method!')
            except BaseException as e1:
                logger.error(e1)
            else:
                logger.info('Successfully using {} to replace the outliers'.format(method1))

        if lower_bound is None and upper_bound is None:  # 判断是否有上下界
            logging.warning('please passing lower_bound or upper_bound')
        else:
            try:
                if lower_bound is None and upper_bound is not None:
                    case = 'case1'
                    s1 = self.s[self.s <= upper_bound]  # 非异常值
                    choose_method(method, case, s1)  # 调用函数替换异常值
                elif lower_bound is not None and upper_bound is None:
                    case = 'case2'
                    s1 = self.s[self.s >= lower_bound]  # 非异常值
                    choose_method(method, case, s1)
                elif lower_bound is not None and upper_bound is not None:
                    case = 'case3'
                    s1 = self.s[(self.s >= lower_bound) & (self.s <= upper_bound)]  # 非异常值
                    choose_method(method, case, s1)
            except BaseException as e:
                logger.error(e)

    def experience_get(self,
                       lower_bound: Optional[Bound] = None,
                       upper_bound: Optional[Bound] = None) -> Series:
        """
        根据经验得到Series的异常值
        :param lower_bound: 下界
        :param upper_bound: 上界
        """
        if lower_bound is None and upper_bound is None:  # 判断是否有上下界
            logging.warning('please passing lower_bound or upper_bound')
        elif lower_bound is not None or upper_bound is not None:
            try:
                if lower_bound is None and upper_bound is not None:
                    return self.s[self.s > upper_bound]  # 返回异常值
                elif lower_bound is not None and upper_bound is None:
                    return self.s[self.s < lower_bound]
                elif lower_bound is not None and upper_bound is not None:
                    return self.s[(self.s < lower_bound) | (self.s > upper_bound)]
            except BaseException as e:
                logger.error(e)
            finally:
                logger.info('Get it. lower_bound is {} and upper_bound is {}'.format(lower_bound, upper_bound))

    def is_normal(self) -> NoReturn:
        """
        用K-S检验判断数据是否为normal distribution
        """
        u = self.s.mean()
        std = self.s.std()
        logger.info("mean:%.2f, standard deviation:%.2f" % (u, std))
        statistic, pvalue = stats.kstest(self.s, 'norm',
                                         (u, std))  # 计算统计量和p值，前者越趋近1越服从正态分布，p值大于0.05则显著服从正态分布正态分布
        logger.info('statistic is {}, pvalue is {}'.format(statistic, pvalue))
        if pvalue > 0.05:  # p值大于0.05，显著，为正态分布
            logger.info('The data follow normal distribution')
        else:
            logger.info('The data do not follow normal distribution')

    def three_sigma_drop(self) -> NoReturn:
        """
        利用3σ准则删除近似正态分布数据的异常值
        """
        try:
            u = self.s.mean()
            std = self.s.std()
            self.s.drop(self.s[(self.s < (u - 3 * std)) | (self.s > (u + 3 * std))].index, inplace=True)  # 删除异常值
            logger.info("Successfully drop outliers with '3σ' method")
        except BaseException as e:
            logger.error(e)

    def three_sigma_replace(self, method: str) -> NoReturn:
        """
        利用3σ准则替换近似正态分布数据的异常值
        :param method: 替换方法
        """
        try:
            u = self.s.mean()
            std = self.s.std()
            if method == 'mean':  # 判断方法
                a = self.s.mean()
                self.s[(self.s < (u - 3 * std)) | (self.s > (u + 3 * std))] = a  # 替换异常值
            elif method == 'median':
                a = self.s.median()
                self.s[(self.s < (u - 3 * std)) | (self.s > (u + 3 * std))] = a
            elif method == 'mode':
                a = self.s.mode()
                self.s[(self.s < (u - 3 * std)) | (self.s > (u + 3 * std))] = a
            elif method == 'bfill':
                self.s[(self.s < (u - 3 * std)) | (self.s > (u + 3 * std))] = np.nan  # 异常值置空
                self.s.fillna(method='bfill', inplace=True)  # 填充置空异常值
            elif method == 'ffill':
                self.s[(self.s < (u - 3 * std)) | (self.s > (u + 3 * std))] = np.nan
                self.s.fillna(method='ffill', inplace=True)
            else:
                raise ValueError('No such method!')
        except BaseException as e:
            logger.error(e)
        else:
            logger.info("Successfully using {} to replace the outliers with '3σ' method".format(method))

    def three_sigma_get(self) -> DataFrame:
        """
        按照3σ法则得到Series的异常值
        """
        try:
            u = self.s.mean()
            std = self.s.std()
            logger.info('Get it. u-3σ is {} and u+3σ is {}'.format(u - 3 * std, u + 3 * std))
            return self.s[(self.s < (u - 3 * std)) | (self.s > (u + 3 * std))]  # 返回异常值
        except BaseException as e:
            logger.error(e)

    def boxplot(self, figsize: Optional[tuple] = None) -> NoReturn:
        """
        画出数据的箱型图
        :param figsize: figure的大小
        """
        plt.figure(figsize=figsize)
        plt.boxplot(self.s, labels=self.s.name)
        plt.show()

    def boxplot_drop(self) -> NoReturn:
        """
        利用箱线图删除数据的异常值
        """
        try:
            q1 = np.percentile(self.s, 25)  # 四分之一分位数
            q3 = np.percentile(self.s, 75)  # 四分之三分位数
            lower_bound = q1 - 1.5 * (q3 - q1)  # 异常值下界
            upper_bound = q3 + 1.5 * (q3 - q1)  # 异常值上界
            self.s.drop(self.s[(self.s < lower_bound) | (self.s > upper_bound)].index, inplace=True)  # 删除异常值
            logger.info("Successfully drop outliers with 'boxplot' method")
        except BaseException as e:
            logger.error(e)

    def boxplot_replace(self, method: str) -> NoReturn:
        """
        利用箱线图法替换数据的异常值
        """
        try:
            q1 = np.percentile(self.s, 25)  # 四分之一分位数
            q3 = np.percentile(self.s, 75)  # 四分之三分位数
            lower_bound = q1 - 1.5 * (q3 - q1)  # 异常值下界
            upper_bound = q3 + 1.5 * (q3 - q1)  # 异常值上界
            if method == 'mean':  # 判断方法
                a = self.s.mean()
                self.s[(self.s < lower_bound) | (self.s > upper_bound)] = a  # 替换异常值
            elif method == 'median':
                a = self.s.median()
                self.s[(self.s < lower_bound) | (self.s > upper_bound)] = a
            elif method == 'mode':
                a = self.s.mode()
                self.s[(self.s < lower_bound) | (self.s > upper_bound)] = a
            elif method == 'bfill':
                self.s[(self.s < lower_bound) | (self.s > upper_bound)] = np.nan  # 异常值置空
                self.s.fillna(method='bfill', inplace=True)  # 填充置空异常值
            elif method == 'ffill':
                self.s[(self.s < lower_bound) | (self.s > upper_bound)] = np.nan
                self.s.fillna(method='ffill', inplace=True)
            else:
                raise ValueError('No such method!')
        except BaseException as e:
            logger.error(e)
        else:
            logger.info("Successfully using {} to replace the outliers with 'boxplot' method".format(method))

    def boxplot_get(self) -> NoReturn:
        """
        按照箱线图法得到某列为异常值的Series
        """
        try:
            q1 = np.percentile(self.s, 25)  # 四分之一分位数
            q3 = np.percentile(self.s, 75)  # 四分之三分位数
            lower_bound = q1 - 1.5 * (q3 - q1)  # 异常值下界
            upper_bound = q3 + 1.5 * (q3 - q1)  # 异常值上界
            logger.info('Get it. lower_bound is {} and upper_bound is {}'.format(lower_bound, upper_bound))
            return self.s[(self.s < lower_bound) | (self.s > upper_bound)]  # 返回异常值
        except BaseException as e:
            logger.error(e)

    # 得到处理后的Series
    def get(self):
        """
        得到处理后的Series
        """
        logger.info('Get cleaned data')
        return self.s


if __name__ == '__main__':
    # 导入数据
    df = pd.read_csv(r'.\FullData.csv')

    # DataCleaning试验
    a = DataCleaning(df)
    # a.drop_null(axis=0, how='any')
    # a.fill_null(National_Kit='China')
    # b = a.get()
    # print(b[b['National_Kit'] == 'China']['National_Kit'])
    a.is_normal(['Long_Shots', 'Curve', 'Penalties'])
    # a.boxplot(['Long_Shots', 'Curve', 'Penalties', 'Speed', 'Balance', 'Strength'], onefig=False)

    # DataCleaningForSeries试验
    s = pd.Series(np.random.normal(10, 5, 100))
    # 老师，这个警告到底是为什么呢?（我查了很久都没搞懂）如果我不用NewType去定义的话，只能都改成np.Series了。如果老师方便回答的话跟我说一下好吗？谢谢！
    c = DataCleaningForSeries(s)
    # c.info()
    # c.drop_null()
    # c.fill_null('mean')
    # c.experience_drop(-10, 10)
    # c.experience_replace('mean', -10, 10)
    # c.experience_get()
    # c.is_normal()
    # c.three_sigma_drop()
    # c.three_sigma_replace('mean')
    # c.three_sigma_get()
    # c.boxplot()
    # c.boxplot_drop()
    # c.boxplot_replace('mean')
    # c.boxplot_get()
    # d = c.get()
    # logger.info('This is cleaned Series:\n {}'.format(d))
