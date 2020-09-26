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
        size_df = fundamentalApi.quotes(asset_code = stock_code, starttime=None, endtime=timepoint_dt-timedelta(days=1), period=1, fields='TotalMv', freq='d', adj=None)
        if(size_df.__len__()==0):
            return np.nan
        else:
            return size_df.values[0][1]

class Illiq(Descriptor):
    '''
    Amihud(2002)的流动性度量指标.
    Illiq = 1/N * sigma_t_N(log(abs(r_t)/Q_t))
    其中N为测量期总日数，r_t为t日收益率，Q_t为t日交易额。

    计算参数：params = {'N':计算Illiq所用回溯期日数}
    '''
    category = {"stock"}

    def calcDayDescriptor(self, stock_code: str, timepoint: str):
        timepoint_dt = datetime.strptime(timepoint, "%Y%m%d")
        try:
            N = self.params['N']
        except Exception as e:
            print("params中必须输入计算非流动性回溯期N!")
            raise(e)
        priceInfo = fundamentalApi.quotes(asset_code=stock_code, starttime=None,
                              endtime=timepoint_dt - timedelta(days=1), period=N, fields=['Open','Close','Amount'], freq='d', adj=None)
        priceInfo['dIlliq'] = abs((priceInfo['Close']/priceInfo['Open'] - 1))/priceInfo['Amount'] * pow(10,8)
        Illiquidity = priceInfo['dIlliq'].mean()
        return Illiquidity

    def calcWeekDescriptor(self, stock_code: str, timepoint: str):
        return self.calcDayDescriptor(stock_code,timepoint)

class BTOP(Descriptor):
    '''
    账面市值比(市净率倒数)
    '''
    category = {"stock"}
    def calcDayDescriptor(self, stock_code: str, timepoint: str):
        timepoint_dt = datetime.strptime(timepoint, "%Y%m%d")
        priceInfo = fundamentalApi.quotes(asset_code=stock_code, starttime=None,
                                          endtime=timepoint_dt - timedelta(days=1), period=1,
                                          fields=['PB'], freq='d', adj=None)
        try:
            BTOP = 1/priceInfo['PB'].values[0]
        except Exception as e:
            BTOP = np.nan
        return BTOP

    def calcWeekDescriptor(self, stock_code: str, timepoint: str):
        BTOP = self.calcDayDescriptor(stock_code, timepoint)
        return BTOP

class PastReturn(Descriptor):
    '''
    过往收益.参数:n1,n2->股票从t-n1-1日起共n2日的累计收益(即t-n1-n2~t-n1-1日的收益)
    '''
    category = {"stock","future"}
    def calcDayDescriptor(self, stock_code: str, timepoint: str):
        timepoint_dt = datetime.strptime(timepoint, "%Y%m%d")
        try:
            n1,n2 = self.params['wait'],self.params['cum']   #间隔期,总长度
            if (type(n1) is not int or type(n2) is not int):
                raise Exception("间隔期wait、区间长度cum必须为非负整数!")
            if(n2<1):
                raise Exception("区间长度cum至少为1!")
        except Exception as e:
            print("参数异常！")
            raise(e)
        priceInfo = fundamentalApi.quotes(asset_code=stock_code, starttime=None,
                                          endtime=timepoint_dt - timedelta(days=1), period=n1+n2,
                                          fields=['Open','Close'], freq='d', adj='hfq')
        try:
            momentum = priceInfo['Close'].values[n1]/priceInfo['Open'].values[n1+n2-1] - 1
        except:
            momentum = np.nan
        return momentum

class FormulaAlpha1(Descriptor):
    '''
    价量背离：短周期内成交量逐步提升，价格不断下降；或成交量逐步下降，价格不断上升
    公式:1/4 *(corr(open,volume)+corr(high,volume)+corr(low,volume)+corr(close,volume))
    参数:回溯期t
    '''
    category = {"stock","future"}
    def calcDayDescriptor(self, stock_code: str, timepoint: str):
        timepoint_dt = datetime.strptime(timepoint, "%Y%m%d")
        try:
            t = self.params['t']  #计算量价背离所用回溯期数
            if (type(t) is not int or t<1):
                raise Exception("回溯期t必须为大于1的整数!")
        except Exception as e:
            print("参数异常！")
            raise(e)
        priceInfo = fundamentalApi.quotes(asset_code=stock_code, starttime=None,
                                          endtime=timepoint_dt - timedelta(days=1), period=t,
                                          fields=['Open','High','Low','Close','Volume'], freq='d', adj='hfq')
        try:
            depart = 0.25*(np.corrcoef(priceInfo['Open'],priceInfo['Volume'])
                             +np.corrcoef(priceInfo['High'],priceInfo['Volume'])
                             +np.corrcoef(priceInfo['Low'],priceInfo['Volume'])
                             +np.corrcoef(priceInfo['Close'],priceInfo['Volume']))[0,1]
        except:
            depart = np.nan
        return depart

class FormulaAlpha2(Descriptor):
    '''
    异常成交量：前一日成交量相对于前一段时间平均成交量的相对变动
    公式:volume/mean(volume)
    参数:用于计算平均成交量的参考期t
    '''
    category = {"stock","future"}
    def calcDayDescriptor(self, stock_code: str, timepoint: str):
        timepoint_dt = datetime.strptime(timepoint, "%Y%m%d")
        try:
            t = self.params['t']  #计算量价背离所用回溯期数
            if (type(t) is not int or t<1):
                raise Exception("回溯期t必须为大于1的整数!")
        except Exception as e:
            print("参数异常！")
            raise(e)
        priceInfo = fundamentalApi.quotes(asset_code=stock_code, starttime=None,
                                          endtime=timepoint_dt - timedelta(days=1), period=t,
                                          fields=['Volume'], freq='d', adj='hfq')
        try:
            abnormalVol = priceInfo['Volume'][0]/np.mean(priceInfo['Volume'])
        except:
            abnormalVol = np.nan
        return abnormalVol

class FormulaAlpha3(Descriptor):
    '''
    量幅背离：短周期内成交量逐步提升，振幅不断下降；或成交量逐步下降，振幅不断提升
    公式:corr(high/low,volume)
    参数:回溯期t
    '''
    category = {"stock","future"}
    def calcDayDescriptor(self, stock_code: str, timepoint: str):
        timepoint_dt = datetime.strptime(timepoint, "%Y%m%d")
        try:
            t = self.params['t']  #计算量价背离所用回溯期数
            if (type(t) is not int or t<1):
                raise Exception("回溯期t必须为大于1的整数!")
        except Exception as e:
            print("参数异常！")
            raise(e)
        priceInfo = fundamentalApi.quotes(asset_code=stock_code, starttime=None,
                                          endtime=timepoint_dt - timedelta(days=1), period=t,
                                          fields=['High','Low','Volume'], freq='d', adj='hfq')
        try:
            depart = np.corrcoef(priceInfo['High']/priceInfo['Low'],priceInfo['Volume'])[0,1]
        except:
            depart = np.nan
        return depart

class NearCloseVol(Descriptor):
    '''
    上个交易日尾盘成交量占比(尾盘n分钟VS整日成交量)
    '''
    category = {"stock"}
    def calcDayDescriptor(self, stock_code: str, timepoint: str):
        try:
            N = self.params['N']
            if(type(N) is not int or N<1 or N>240):
                raise("N必须为0~240之间的正整数!")
        except Exception as e:
            print("params中必须输入计算获取尾盘分钟数N!")
            raise(e)
        endtime = datetime.strptime(timepoint,"%Y%m%d")
        df_dayminute = fundamentalApi.quoteMinute(stock_code=stock_code,starttime=None,endtime=endtime,period=240,fields='Volume')
        ratio = df_dayminute["Volume"].iloc[0:N].sum()/df_dayminute["Volume"].sum()
        return ratio

class FomulaDailyFactor(Descriptor):
    '''
    公式因子，用于迭代.根据输入的公式树结构以及预先定义的算子,确定计算因子的公式
    todo 建设中
    '''
    #算子定义区
    @classmethod
    def solveTree(cls,formulaTree:dict,stock_code:str,timepoint:str,n=0):
        '''
        中间节点:
            普通算子:max,corr,log,delay,...
        叶节点:
            特殊算子:close,open,high,low,volumn...
            常数算子:const
        '''
        func,params = formulaTree.popitem()  #func必定为算子,params在func为中间节点时为字典{'func1':{},'func2':{},'const':6},叶节点时func='const'时为常数,func='OPEN'等数据项时为'null'
        if(func=='const'):
            return params
        if(func in ('OPEN','CLOSE','HIGH','LOW','VOLUME')):
            #todo 用stock_code,timepoint,n在内存中取数
            return 0
        if(func=='max'):
            if(params.__len__()!=2):
                raise Exception('max有2个参数!')
            calcedRst = [cls.solveTree({subfunc:subparam},stock_code,timepoint,n) for subfunc,subparam in params.items()]
            return np.max(calcedRst)
        if(func=='corr'):
            if(params.__len__()!=3):
                raise Exception('corr有3个参数!')
            subfunc1,subparam1 = params.popitem()
            subfunc2,subparam2 = params.popitem()
            useless,period = params.popitem()
            series1 = [cls.solveTree({subfunc1:subparam1},stock_code,timepoint,n+i) for i in range(period)]
            series2 = [cls.solveTree({subfunc2:subparam2},stock_code,timepoint,n+i) for i in range(period)]
            return np.corrcoef(series1,series2)[0,1]


    category = {"stock","future"}
    def calcDayDescriptor(self, stock_code: str, timepoint: str):
        pass