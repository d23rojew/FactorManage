'''
author: dongruoyu
time: 2020-02-10
提供便捷获取基础数据的api
'''
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from dateutil.parser import parse
from time import time
from typing import Union,List
import os

dbPath = os.path.dirname(__file__) + '\\testdb'
def outertrade_cal(conn:sqlite3.Connection, exchange: str, startdate: str, enddate: str, is_open: str = None) -> pd.DataFrame:
    '''
    取交易日
    '''
    sqlstr = "select * from TradingDays where exchange = '{exchange}' and date between '{startdate}' and '{enddate}'".format(
        exchange=exchange, startdate=startdate, enddate=enddate)
    if (is_open != None):
        sqlstr += " and is_open = '{is_open}'".format(is_open=is_open)
    sqlstr += " order by exchange asc,date asc"
    df = pd.read_sql_query(sqlstr, conn)
    return df

df_tradingdate = outertrade_cal(sqlite3.Connection(dbPath),exchange='SSE', startdate='19000101', enddate='21000101', is_open='1')
df_tradingdate['dateIndex'] = pd.to_datetime(df_tradingdate['date'], format='%Y%m%d')
df_tradingdate.set_index('dateIndex', inplace=True)
df_tradingdate.index = pd.DatetimeIndex(df_tradingdate.index)

class fundamentalApi:
    conn = sqlite3.Connection(dbPath)
    @classmethod
    def trade_cal(cls,exchange:str,startdate:str,enddate:str,is_open:str='1')->pd.DataFrame:
        '''
        获取交易日
        :param exchange: 'SSE' 上交所, 'SZSE' 深交所, 'CFFEX' 中金所 ,'SHFE' 上期所 ,'CZCE' 郑商所 ,'DCE' 大商所,'INE' 上能源 ,'IB' 银行间 ,'XHKG' 港交所
        :param startdate: 起始日期,形如'20151231'
        :param enddate: 终止日期,形如'20151231'
        :param is_open: 开市'1' 休市'0'
        :return: DataFrame,cols:exchange,date,is_open
        '''
        return outertrade_cal(cls.conn, exchange, startdate, enddate, is_open)

    classmethod
    def get_trade_date(cls,baseDate:str,shift:int)->str:
        '''
        用给定日期以及偏离日计算交易日
        :param baseDate: 制定日期
        :param shift: 偏离日
        :return: 交易日'yyyymmdd'
        '''
        dateSeries = df_tradingdate['date']
        seq = (dateSeries[dateSeries <= baseDate]).__len__() - 1 + shift
        datelist = list(dateSeries)
        if (seq <= 0):
            seq = 0
        if (seq >= datelist.__len__()-1):
            seq = datelist.__len__()-1
        return datelist[seq]

    __quotes_facilitate_dateDict = {}
    @classmethod
    def quotes(cls, asset_code:str, starttime:datetime, endtime:datetime, period:int, fields:Union[str,List[str]], freq:str, adj:str, asset_type:str = 'stock')->pd.DataFrame:
        '''
        获取价格序列数据
        :param asset_code:股票代码(股票代码列表)OR指数代码(指数代码列表)
        :param starttime:开始时间(period/starttime任填一个)
        :param endtime:结束时间
        :param period:获取的价格区间总长度(period/starttime任填一个)
        :param field:Open/High/Low/Close/TotalMV/...(见DBdefinition.xml)
        :param freq:'min'(分钟)/'day'(日)
        :param adj:None(未复权)/'qfq'(前复权)/'hfq'(后复权)
        :param asset_type:'stock'(股票)/'index'(指数)
        :return:以stock_code\time为主键,其它价格为值域的DataFrame
        '''
        TableFields = ''
        AssetCodestr = ''
        if(type(fields)==str):
            fields = [fields]
        if(asset_type not in ('stock','index')):
            raise Exception("不支持的资产类型"+asset_type+",当前支持资产类型为'stock'或'index'")
        elif(asset_type == 'stock'):
            CodeFieldName = 'StockCode'
            TableName = 'StockMarketDaily'
        elif(asset_type == 'index'):
            CodeFieldName = 'IndexCode'
            TableName = 'IndexMarketDaily'
        for fld in fields:
            if(fld in ('Open','High','Low','Close') and adj is not None and asset_type == 'stock'):
                if (adj == 'hfq'):
                    TableFields += ','+fld+'*Adjfactor '+fld
                elif (adj == 'qfq'):
                    raise Exception('行情接口price()暂不支持前复权价格数据!')
                else:
                    raise Exception("行情接口price参数异常:adj输入了未预期到的参数" + str(adj))
            else:
                TableFields += ' ,'+fld
        if(asset_code is None):
            AssetCodeControl = ""
        elif(type(asset_code) is str):
            AssetCodeControl =  "and {CodeFieldName} in ('{asset_code}')".format(CodeFieldName=CodeFieldName, asset_code = asset_code)
        elif(type(asset_code) is not list):
            raise Exception("参数asset_code需为字符串或字符串列表,当前类型为{type}".format(str(type(asset_code))))
        else:
            for code in asset_code:
                AssetCodestr += ",'" + code + "'"
            AssetCodestr = AssetCodestr.strip(',')
            AssetCodeControl = "and {CodeFieldName} in ({AssetCodestr})".format(CodeFieldName=CodeFieldName, AssetCodestr = AssetCodestr)
        endTimestr = endtime.strftime("%Y%m%d")
        if(starttime is None and period is not None):
            startTimestr = cls.get_trade_date(cls,endTimestr,-period+1)
        else:
            startTimestr = starttime.strftime("%Y%m%d")
        sqlstr = "select Time,{CodeFieldName} {TableFields} from {TableName} where Time>='{starttime}' and Time<='{endtime}' {AssetCodeControl} order by {CodeFieldName} asc,Time desc "\
            .format(AssetCodeControl=AssetCodeControl, TableFields=TableFields,starttime=startTimestr,endtime=endTimestr,CodeFieldName=CodeFieldName,TableName=TableName)
        df = pd.read_sql_query(sqlstr,cls.conn,index_col='Time')
        df.replace(to_replace=[None],value=np.nan,inplace=True)
        df.index = pd.DatetimeIndex(df.index.astype(str))
        return df

    __quotes_facilitate_MinuteDict = {}
    @classmethod
    def quoteMinute(cls, asset_code:str, starttime:datetime, endtime:datetime, period:int=None, fields:Union[str,List[str]]=['Open'])->pd.DataFrame:
        '''
        获取分钟价格序列数据
        :param asset_code:股票代码(股票代码列表)
        :param starttime:开始时间(period/starttime任填一个)
        :param endtime:结束时间
        :param period:获取的价格区间总长度(period/starttime任填一个)
        :param field:Open/High/Low/Close/Volume/Amount(见DBdefinition.xml)
        :param adj:'None'(未复权)/'qfq'(前复权)/'hfq'(后复权)
        :return:以stock_code\time为主键,其它价格为值域的DataFrame
        '''
        TableFields = ''
        AssetCodestr = ''
        if(type(fields)==str):
            fields = [fields]
        for fld in fields:
            TableFields += ' ,'+fld
        if(asset_code is None):
            AssetCodeControl = ""
        elif(type(asset_code) is str):
            AssetCodeControl =  "and StockCode in ('{asset_code}')".format(asset_code = asset_code)
        elif(type(asset_code) is not list):
            raise Exception("参数asset_code需为字符串或字符串列表,当前类型为{type}".format(str(type(asset_code))))
        else:
            for code in asset_code:
                AssetCodestr += ",'" + code + "'"
            AssetCodestr = AssetCodestr.strip(',')
            AssetCodeControl = "and StockCode in ({AssetCodestr})".format(AssetCodestr = AssetCodestr)
        endTimestr = endtime.strftime("%Y-%m-%d %H:%M:%S")
        if(starttime is None and period is not None):
            startTimestr = cls.__quotes_facilitate_MinuteDict.get((endTimestr,period))
            if(startTimestr is None):
                df_minutes = cls.TradingTimePoints(asset_code=None,starttime=datetime.strptime('20100101','%Y%m%d'),endtime=endtime,freq='min')
                startIndex = 0 if df_minutes.__len__() - period < 0 else df_minutes.__len__() - period
                startTimestr = str(df_minutes.index[startIndex])
                cls.__quotes_facilitate_MinuteDict[(endTimestr,period)] = startTimestr
        else:
            startTimestr = starttime.strftime("%Y-%m-%d %H:%M:%S")
        sqlstr = "select Time,StockCode {TableFields} from StockMarketMinute where Time>='{starttime}' and Time<='{endtime}' {AssetCodeControl} order by StockCode asc,Time desc "\
            .format(AssetCodeControl=AssetCodeControl, TableFields=TableFields,starttime=startTimestr,endtime=endTimestr)
        df = pd.read_sql_query(sqlstr,cls.conn,index_col='Time')
        df.replace(to_replace=[None],value=np.nan,inplace=True)
        df.index = pd.DatetimeIndex(df.index.astype(str))
        return df

    @classmethod
    def getReturn(cls, asset_code:str, starttime:datetime, endtime:datetime, forcast_period:int,freq:str = 'd',asset_type:str = 'stock'):
        '''
        获取资产在给定时间区间内的收益．该收益在开始时间一定可交易，否则收益为nan
        :param asset_code: 资产代码
        :param starttime: 起始时间
        :param endtime: 结束时间
        :param forcast_period: 预测期
        :param freq: 频率'min'/'d'/'w'/'m'
        :return:dataframe，横轴为资产，纵轴为时间
        '''
        if(freq=='d'):
            endtimestr = endtime.strftime('%Y%m%d')
            quoteendtimestr = cls.get_trade_date(cls,endtimestr,forcast_period-1)
            quoteendtime = datetime.strptime(quoteendtimestr,'%Y%m%d')
        priceInfo = cls.quotes(asset_code,starttime,quoteendtime,period=None,fields=['Open','High','Low','Close'],freq=freq,adj='hfq',asset_type=asset_type)
        filtSeries = (priceInfo['Open']==priceInfo['High']).astype(bool) &(priceInfo['Open']==priceInfo['Low']).astype(bool) &(priceInfo['Open']==priceInfo['Close']).astype(bool)
        priceInfo.loc[filtSeries,'Open'] = np.nan  #在开仓时必须为正常行情，否则剔出样本
        panels = priceInfo.pivot(columns='StockCode',values=['Open','Close'])
        if(panels.__len__()==0):
            return pd.DataFrame()
        returnMat = panels['Close'].shift(1-forcast_period)/panels['Open'] - 1
        return returnMat[0:returnMat.__len__()-forcast_period+1]

    Tdate = {freq: df_tradingdate.resample(freq, convention='start').first()['date'].dropna() for freq in ['d', 'w', 'm']}
    timepoint = Tdate['d'].asfreq('min')
    t1 = time()
    Tminute = None
    @classmethod
    def __preGetminute(cls):
        if(cls.Tminute is None):
            tmin = cls.timepoint.index.strftime('%H%M')
            cls.Tminute = cls.timepoint[((tmin >= '0931') & (tmin <= '1130'))
                                |((tmin >= '1301') & (tmin <= '1500'))]
        return cls.Tminute

    @classmethod
    def TradingTimePoints(cls,asset_code:str,starttime:datetime,endtime:datetime,freq:str) -> pd.DataFrame:
        '''
        获取所选资产的交易时间
        :param asset_code: 资产代码，可为具体代码(如股票'000001.SZ')或资产类别(如'stock')
        :param starttime: 开始时间点
        :param endtime: 结束时间点
        :param freq: 频率，可为(min/d/w/m)
        :return: 可交易时间点的datetime序列(freq为m则返回可交易分钟序列；其它则返回每个可交易周期的第一个交易日)
        todo 目前freq为'min'时仅返回a股可交易分钟序列。期货每日可交易时间各品种、各合约都可能不同，需要后续单独建设
        '''
        if(freq in ['d','w','m']):
            dates = cls.Tdate[freq]
            return dates[(dates >= starttime.strftime('%Y%m%d')) & (dates <= endtime.strftime('%Y%m%d'))]
        if(freq == 'min'):
            Tminute = cls.__preGetminute()
            tradingTimepoint = Tminute[(starttime<=cls.Tminute.index) & (endtime>=cls.Tminute.index)]
            return tradingTimepoint

    @classmethod
    def stockinfo(cls, exchange:list=None, list_status:list=None)->pd.DataFrame:
        '''
        根据交易所/上市状况查询股票名单及明细情况
        :param exchange:筛选条件【交易所】,str OR list,可选"SSE"(上交所)/"SZSE"(深交所)
        :param list_status:筛选条件【上市状况】,str OR list,可选"上市"/"退市"/"暂停上市"
        :return:以fields为列，StockCode为索引的DataFrame
                  fields:  name          股票名称
                           area          所在地域
                           industry      所属行业
                           fullname      股票全称
                           enname        英文全称
                           market        市场类型
                           exchange      交易所
                           curr_type     交易货币
                           list_status   上市状态
                           list_date     上市日期
                           delist_date   退市日期
                           is_hs         是否沪深港通标的
        '''
        exchangeCondition = ""
        liststatusCondition = ""
        if(exchange is not None):
            if(type(exchange) is str):
                exchangeCondition = "and exchange = '{exchangeStr}'".format(exchangeStr=exchange)
            elif(type(exchange) is list):
                for exc in exchange:
                    exchangeCondition += "'"+exc+"',"
                exchangeCondition = "and exchange in (" + exchangeCondition.strip(',') + ")"
        if(list_status is not None):
            if(type(list_status) is str):
                liststatusCondition = "and list_status = '{liststatusStr}'".format(liststatusStr = list_status)
            elif(type(list_status) is list):
                for ls in list_status:
                    liststatusCondition += "'"+ls+"',"
                liststatusCondition = "and list_status in (" + liststatusCondition.strip(',') + ")"

        sql = "select * from StockBasics where 1=1 {exchangeCondition} {liststatusCondition}".format(exchangeCondition=exchangeCondition,
                                                                                                      liststatusCondition=liststatusCondition)
        df = pd.read_sql_query(sql,cls.conn,index_col='StockCode')
        return df


