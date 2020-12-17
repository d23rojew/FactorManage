'''
择时模型:
根据选择的资产、特征，在样本时间区间内训练择时模型，并进行样本外回测.
输入:资产代码、特征对象列表、样本时间区间；
输出:一个可给出回测报告(输入时间区间即可)的对象。

对其它api的假定:
get_feature(descriptor:Descriptor,starttime:datetime, endtime:datetime,stock_name:list=None,check:bool=True)
descriptor的频率交由get_feature来指定，并在对象初始化中被剔除。

todo:现在回头补写一些基础数据和因子特征算法，用multifactor模型研究一下因子收益的持续性；
     加入残差动量因子:将因子横街面回归残差
'''
from typing import List,Union
from datetime import datetime
from core.features.Descriptor import Descriptor
from core.fundamentals.getfundamentals import fundamentalApi as fapi
from core.features.getfeature import get_feature
class TSmodel:
    def __init__(self,asset_code:str,factors:List[Union[Descriptor]],starttime:datetime,endtime:datetime,freq:str='d',forcast_period:int=1):
        '''
        :param stockcode: 待预测标的
        :param factors: 特征列表
        :param starttime: 样本起始时间
        :param endtime: 样本结束时间
        :param freq: 学习频率 'm'(分)/'d'(日)/'w'(周)/'M'(月)
        :param forcast_period: 预测期
        '''
        #todo 如何预测同标的的期货合约(要考虑合约换月)？
        #     (丢给feature算法/get_return特殊处理，此处仅考虑可以由stockcode顺利获取特征及收益)
        ReturnSeries = fapi.getReturn(asset_code=asset_code,starttime=starttime, endtime=endtime,forcast_period=forcast_period,freq=freq)
        for factor in factors:
            factor.freq = freq
            feSeries = get_feature(factor,starttime, endtime,asset_code,1)


    def fit(self,starttime:datetime,endtime:datetime):
        '''
        使用已拟合的模型进行预测
        :param starttime: 预测起始时间
        :param endtime:  预测结束时间
        :return: 输出预测的收益序列、波动率及真实收益(如果有的话)
        '''
        pass

    def save(self):
        '''
        存储模型及报告
        :return:
        '''
        pass