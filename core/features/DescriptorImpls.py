'''
author: dongruoyu
time: 2020-03-16
特征的实现类
'''
from core.features.Descriptor import Descriptor
from core.fundamentals.getfundamentals import fundamentalApi
import statsmodels.api as sm
import numpy as np
from datetime import datetime
from datetime import timedelta
import pandas as pd
from collections import OrderedDict

class templatefeature(Descriptor):
    '''
    一个特征计算类的样例
    '''
    @classmethod
    def calcDescriptor(cls, stock_name:str, timepoint:str):
        return timepoint

class CapitalGainOverhang(Descriptor):
    '''
    技术指标: 前日未实现损益(CapitalGainOverhang)
    k日的【前日未实现损益】指k - 1
    日收盘时市场平均未实现损益.
    参数:N:计算CGO所回溯的交易日
    '''

    category = {"stock","commodity"}

    def calcDayDescriptor(self, stock_code:str, timepoint:str):
        try:
            period = self.params['N']
        except Exception as e:
            raise("计算平均持股成本缺少回溯期数N")
        timepoint = datetime.strptime(timepoint,'%Y%m%d')
        priceInfo = fundamentalApi.quotes(asset_code=stock_code, starttime=None, endtime=timepoint - timedelta(days=1), period=period, fields=['Open', 'High', 'Low', 'Close', 'Tor_f'], freq='d', adj='hfq')
        if((priceInfo['Tor_f']>0).sum()<period*2/3 or priceInfo.__len__()==0):
            cgoValue = np.nan
        else:
            yesterdayClose = priceInfo['Close'].values[0]
            priceInfo['Tor_f'] = priceInfo['Tor_f'].clip(upper=1) #换手率超过100%的视为100%
            priceInfo.sort_index(ascending=False,inplace=True)
            priceInfo['remianPart'] = (1-priceInfo['Tor_f']).shift(1).fillna(1).cumprod()
            priceInfo['relativeWeight'] = priceInfo['remianPart'] * priceInfo['Tor_f']
            priceInfo['priceWeight'] = priceInfo['relativeWeight']/priceInfo['relativeWeight'].sum()
            priceInfo['avgPrice'] = (priceInfo['High'] + priceInfo['Low'] + priceInfo['Open'] + priceInfo['Close'])/4
            referencePrice = (priceInfo['priceWeight'] * priceInfo['avgPrice']).sum()
            cgoValue = yesterdayClose/referencePrice - 1
        return cgoValue

class Beta(Descriptor):
    '''
    股票收益与指定指数的Beta
    参数：N:int,计算Beta所回溯的交易日
         index:str,指定的指数(hs300)
    '''

    category = {"stock"}

    def calcWeekDescriptor(self, stock_code: str, timepoint: str):
        try:
            period = self.params['period']
            indexCode = self.params['index']
        except Exception as e:
            raise ("计算股票beta缺少回溯期数period或参照指数index")
        timepoint = datetime.strptime(timepoint, '%Y%m%d')
        stockReturn = fundamentalApi.quotes(asset_code=stock_code, starttime=None, endtime=timepoint - timedelta(days=1),
                                          period=period, fields=['Close'], freq='d',
                                          adj='hfq')['Close'].pct_change()
        indexReturn = fundamentalApi.quotes(asset_code=indexCode, starttime=None, endtime=timepoint - timedelta(days=1),
                                          period=period, fields=['Close'], freq='d',
                                          adj=None,asset_type='index')['Close'].pct_change()
        mergeReturn = pd.DataFrame({'stock':stockReturn,'index':indexReturn})
        try:
            model = sm.OLS(mergeReturn['stock'].values.reshape(-1,1),sm.add_constant(mergeReturn['index'].values.reshape(-1, 1)), missing='drop')
            rst = model.fit()
            Beta = rst.params[1]
        except Exception as e:
            Beta = np.nan
        return Beta

class Size(Descriptor):
    '''
    总市值(元)
    '''
    category = {"stock"}

    def calcWeekDescriptor(self, stock_code: str, timepoint: str):
        timepoint_dt = datetime.strptime(timepoint,"%Y%m%d")
        size_df = fundamentalApi.quotes(asset_code = stock_code, starttime=timepoint_dt, endtime=timepoint_dt, period=1, fields='TotalMv', freq='d', adj=None)
        if(size_df.__len__()==0):
            return np.nan
        else:
            return size_df.values[0][1]