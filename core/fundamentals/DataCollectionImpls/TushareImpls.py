import pandas as pd
import datetime
import sqlite3
import tushare as ts
import time
from tqdm import tqdm
from core.fundamentals.DataCollectionImpls.DataCollection import DataCollection
from core.fundamentals.getfundamentals import fundamentalApi as fapi

ts.set_token('46be315f285d45d0db65eb8cba19b61a6b90599169230ed2b83d2470')
api = ts.pro_api()

class CollectTradingDay(DataCollection):
    '''
    从tushare.trade_cal获取交易日日历
    '''
    exchangeList = ['SSE'  #上交所
                    ,'SZSE' #深交所
                    ,'CFFEX' #中金所
                    ,'SHFE' #上期所
                    ,'CZCE' #郑商所
                    ,'DCE' #大商所
                    ,'INE' #上能源
                    ,'IB' #银行间
                    ,'XHKG' #港交所
                    ]

    @classmethod
    def renew(cls,conn:sqlite3.Connection):
        print('正在从tushare更新交易日历...')
        try:
            cls.get_data(StockCode=None,starttime=datetime.datetime.now()-datetime.timedelta(weeks=520),endtime=datetime.datetime.now()+datetime.timedelta(weeks=52),conn=conn)
            print('交易日历更新完毕！')
        except Exception as e:
            print("通过Tushare更新交易日历时出现错误:"+str(e.args))

    @classmethod
    def apidata(self, StockCode:str, starttime:datetime, endtime:datetime)->pd.DataFrame:
        '''
        输入参数:starttime,endtime
        总是从tushare->trade_cal获取给定时间段内各个交易所的开、休市情况
        '''
        calendar_df = pd.DataFrame()
        for exchange in self.exchangeList:
            calendar_df = calendar_df.append(api.trade_cal(exchange=exchange
                                                           ,start_date=starttime.strftime('%Y-%m-%d')
                                                           ,end_date=endtime.strftime('%Y-%m-%d')))
        return calendar_df

    mapper = {'TradingDays':{'exchange':'exchange','date':'cal_date','is_open':'is_open'}}

class CollectStockDailyPrice(DataCollection):
    '''
    从tushare.trade_cal获取日行情数据
    '''
    @classmethod
    def apidata(cls, StockCode: str, starttime: datetime, endtime: datetime) -> pd.DataFrame:
        '''
        输入参数:starttime,endtime
        从tushare->daily接口中获取所有股票在某时间段内的行情
        .daily接口每分钟允许调用500次
        '''
        trade_date = fapi.trade_cal('SSE',starttime.strftime('%Y%m%d'),endtime.strftime('%Y%m%d'),'1')
        price_df = pd.DataFrame()
        for date in tqdm(trade_date["date"],desc='正在更新...',ncols=80):
            df = None
            num = 0
            while(df is None):
                try:
                    df = api.daily(trade_date=date)
                    num=0
                except Exception as e:
                    if(num<5):
                        num+=1
                        print(e)
                        print('通过tushare获取{date}日行情时出现异常，将进行第{num}次重试'.format(date=date,num=str(num)))
                    else:
                        print('更新日行情失败!')
                        raise e
            price_df = price_df.append(df)
        price_df["amount"] = price_df["amount"] * 1000; #tushare的交易额单位为(千元)
        return price_df

    @classmethod
    def renew(cls,conn:sqlite3.Connection):
        trade_date = fapi.trade_cal('SSE', '20180101', datetime.datetime.now().strftime('%Y%m%d'),'1')
        MissingDates = [];
        lastDate = trade_date["date"].tolist()[-1]
        beginDate, endDate = None, None
        timespan = 0
        for date in tqdm(trade_date["date"],desc='正在自检日度行情...',ncols=80):
            is_exist = cls.probe(conn,trade_date=date)
            if(not is_exist):
                timespan += 1
                endDate=date
                if(not beginDate):
                    beginDate=date
                if(date == lastDate or timespan>100):
                    MissingDates.append((beginDate, endDate))
                    timespan = 0
                    beginDate, endDate = None, None
            elif(beginDate):
                    MissingDates.append((beginDate,endDate))
                    timespan = 0
                    beginDate,endDate = None,None
        if(MissingDates.__len__()>0):
            for bgdt,eddt in tqdm(MissingDates,desc='正在从tushare更新日度行情...',ncols=90):
                cls.get_data('',datetime.datetime.strptime(bgdt,'%Y%m%d')
                             ,datetime.datetime.strptime(eddt,'%Y%m%d')
                             ,conn)
        print('日度行情更新完毕！')
    mapper = {'StockMarketDaily':{'StockCode':'ts_code','Time':'trade_date','Open':'open','High':'high','Low':'low','Close':'close','Volume':'vol','Amount':'amount'}}

class CollectAdjFactor(DataCollection):
    '''
    从tushare.adj_factor获取复权因子数据
    '''
    @classmethod
    def apidata(cls, StockCode: str, starttime: datetime, endtime: datetime) -> pd.DataFrame:
        trade_date = fapi.trade_cal('SSE',starttime.strftime('%Y%m%d'),endtime.strftime('%Y%m%d'),'1')
        adjfactor_df = pd.DataFrame()
        for date in tqdm(trade_date["date"],desc='正在更新...',ncols=80):
            df = None
            num = 0
            while(df is None):
                try:
                    df = api.adj_factor(trade_date=date)
                    num=0
                except Exception as e:
                    if(num<5):
                        num+=1
                        print(e)
                        print('通过tushare获取{date}日行情时出现异常，将进行第{num}次重试'.format(date=date,num=str(num)))
                    else:
                        print('更新日行情失败!')
                        raise e
            adjfactor_df = adjfactor_df.append(df)
        return adjfactor_df

    @classmethod
    def renew(cls,conn:sqlite3.Connection):
        trade_date = fapi.trade_cal('SSE', '20180101', datetime.datetime.now().strftime('%Y%m%d'),'1')
        lastDate = trade_date["date"].tolist()[-1]
        trade_date.rename(columns = {'date':'trade_date'},inplace=True)
        print('正在自检复权因子...')
        lacks = set(cls.probe(conn,trade_date[['trade_date']])['trade_date'])
        MissingDates = [];
        beginDate, endDate = None, None
        timespan = 0
        for date in trade_date["trade_date"]:
            is_exist = date not in lacks
            if(not is_exist):
                timespan += 1
                endDate=date
                if(not beginDate):
                    beginDate=date
                if(date == lastDate or timespan>200):
                    MissingDates.append((beginDate, endDate))
                    timespan = 0
                    beginDate, endDate = None, None
            elif(beginDate):
                    MissingDates.append((beginDate,endDate))
                    timespan = 0
                    beginDate,endDate = None,None
        if(MissingDates.__len__()>0):
            for bgdt,eddt in tqdm(MissingDates,desc='正在从tushare更新复权因子...',ncols=90):
                cls.get_data('',datetime.datetime.strptime(bgdt,'%Y%m%d')
                             ,datetime.datetime.strptime(eddt,'%Y%m%d')
                             ,conn)
        print('复权因子更新完毕！')

    mapper = {'StockMarketDaily':{'StockCode':'ts_code','Time':'trade_date','Adjfactor':'adj_factor'}}

class CollectDailyBasics(DataCollection):
    '''
    从tushare.daily_basic获取股票每日指标。包括:
    turnover_rate	float	换手率（%）
    turnover_rate_f	float	换手率（自由流通股）
    volume_ratio	float	量比
    pe	            float	市盈率（总市值/净利润）
    pe_ttm	        float	市盈率（TTM）
    pb	            float	市净率（总市值/净资产）
    ps	            float	市销率
    ps_ttm	        float	市销率（TTM）
    dv_ratio	    float	股息率 （%）
    dv_ttm	        float	股息率（TTM）（%）
    total_share	    float	总股本 （万股）
    float_share	    float	流通股本 （万股）
    free_share	    float	自由流通股本 （万）
    total_mv	    float	总市值 （万元）
    circ_mv	        float	流通市值（万元）
    '''
    @classmethod
    def apidata(cls, StockCode: str, starttime: datetime, endtime: datetime) -> pd.DataFrame:
        trade_date = fapi.trade_cal('SSE',starttime.strftime('%Y%m%d'),endtime.strftime('%Y%m%d'),'1')
        daily_basic = pd.DataFrame()
        for date in tqdm(trade_date["date"],desc='正在更新...',ncols=80):
            df = None
            num = 0
            while(df is None):
                try:
                    df = api.daily_basic(trade_date=date)
                    num=0
                except Exception as e:
                    if(num<5):
                        num+=1
                        print(e)
                        print('通过tushare获取{date}日股票基本面指标时出现异常，将进行第{num}次重试'.format(date=date,num=str(num)))
                    else:
                        print('更新日行情失败!')
                        raise e
            daily_basic = daily_basic.append(df)
        daily_basic["turnover_rate"] = daily_basic["turnover_rate"] / 100
        daily_basic["turnover_rate_f"] = daily_basic["turnover_rate_f"] / 100
        daily_basic["dv_ratio"] = daily_basic["dv_ratio"] / 100
        daily_basic["dv_ttm"] = daily_basic["dv_ttm"] / 100
        daily_basic["total_share"] = daily_basic["total_share"] * 10000
        daily_basic["float_share"] = daily_basic["float_share"] * 10000
        daily_basic["free_share"] = daily_basic["free_share"] * 10000
        daily_basic["total_mv"] = daily_basic["total_mv"] * 10000
        daily_basic["circ_mv"] = daily_basic["circ_mv"] * 10000
        return daily_basic

    @classmethod
    def renew(cls,conn:sqlite3.Connection):
        trade_date = fapi.trade_cal('SSE', '20180101', datetime.datetime.now().strftime('%Y%m%d'),'1')
        lastDate = trade_date["date"].tolist()[-1]
        trade_date.rename(columns = {'date':'trade_date'},inplace=True)
        print('正在自检股票每日基本面指标...')
        lacks = set(cls.probe(conn,trade_date[['trade_date']])['trade_date'])
        MissingDates = [];
        beginDate, endDate = None, None
        timespan = 0
        for date in trade_date["trade_date"]:
            is_exist = date not in lacks
            if(not is_exist):
                timespan += 1
                endDate=date
                if(not beginDate):
                    beginDate=date
                if(date == lastDate or timespan>200):
                    MissingDates.append((beginDate, endDate))
                    timespan = 0
                    beginDate, endDate = None, None
            elif(beginDate):
                    MissingDates.append((beginDate,endDate))
                    timespan = 0
                    beginDate,endDate = None,None
        if(MissingDates.__len__()>0):
            for bgdt,eddt in tqdm(MissingDates,desc='正在从tushare更新股票基本面指标...',ncols=90):
                cls.get_data('',datetime.datetime.strptime(bgdt,'%Y%m%d')
                             ,datetime.datetime.strptime(eddt,'%Y%m%d')
                             ,conn)
        print('基本面指标更新完毕！')

    mapper = {'StockMarketDaily':{'StockCode':'ts_code','Time':'trade_date','Tor':'turnover_rate','Tor_f':'turnover_rate_f'
                                  ,'VolumeRatio':'volume_ratio','PE_lyr':'pe','PE_ttm':'pe_ttm','PB':'pb','PS':'ps','PS_ttm':'ps_ttm'
                                  ,'DV_ratio':'dv_ratio','DV_ratio_ttm':'dv_ttm','TotalShare':'total_share'
                                  ,'FloatShare':'float_share','FreeShare':'free_share','TotalMV':'total_mv','CircMV':'circ_mv'}}

class CollectStockBasics(DataCollection):
    '''
    从tushare.stock_basic获取股票基本指标。包括:
    StockCode    str	  股票代码
    Name	     str	  股票名称
    Area	     str	  所在地域
    industry     str	  所属行业
    fullname     str	  股票全称
    enname	     str	  英文全称
    market	     str	  所属市场(主板/中小板/创业板/科创板)
    exchange     str      交易所代码
    curr_type    str	  交易货币
    list_status  str	  上市状态(L上市 D退市 P暂停上市")
    list_date	 str      上市日期
    delist_date	 str   	  退市日期
    is_hs	     str      是否沪深港通标的(N否/H沪股通/S深股通)
    '''
    @classmethod
    def apidata(cls, StockCode: str, starttime: datetime, endtime: datetime) -> pd.DataFrame:
        daily_basic_l = api.stock_basic(list_status='L',fields='ts_code,name,area,industry,fullname,enname,market,exchange,curr_type,list_status,list_date,delist_date,is_hs')
        daily_basic_d = api.stock_basic(list_status='D',fields='ts_code,name,area,industry,fullname,enname,market,exchange,curr_type,list_status,list_date,delist_date,is_hs')
        daily_basic_p = api.stock_basic(list_status='P',fields='ts_code,name,area,industry,fullname,enname,market,exchange,curr_type,list_status,list_date,delist_date,is_hs')
        daily_basic = daily_basic_l.append([daily_basic_d,daily_basic_p])
        daily_basic["list_status"].replace({"L":"上市","D":"退市","P":"暂停上市"},inplace=True)
        daily_basic["is_hs"].replace({"N": "否", "H": "沪股通", "S": "深股通"}, inplace=True)
        return daily_basic

    @classmethod
    def renew(cls,conn:sqlite3.Connection):
        print('正在更新股票基本信息...')
        cls.get_data(None,None,None,conn)
        print('股票基本信息更新完毕！')

    mapper = {'StockBasics':{'StockCode':'ts_code','Name':'name','Area':'area','industry':'industry'
                                  ,'fullname':'fullname','enname':'enname','market':'market','exchange':'exchange','curr_type':'curr_type','list_status':'list_status'
                                  ,'list_date':'list_date','delist_date':'delist_date','is_hs':'is_hs'}}

class CollectHS300Daily(DataCollection):
    '''
    从tushare.index_daily获取沪深300指数日行情数据
    '''
    @classmethod
    def apidata(cls, StockCode: str, starttime: datetime, endtime: datetime) -> pd.DataFrame:
        '''
        输入参数:starttime,endtime
        从tushare->index_daily接口中获取给定指数在某时间段内的行情
        '''
        start_date = starttime.strftime('%Y%m%d') if starttime is not None else None
        end_date = endtime.strftime('%Y%m%d') if endtime is not None else None
        df = None
        num = 0
        while(df is None):
            try:
                df = api.index_daily(ts_code = StockCode,start_date = start_date,end_date = end_date)
                num=0
            except Exception as e:
                if(num<5):
                    num+=1
                    print(e)
                    print('通过tushare获取指数行情时出现异常，将进行第{num}次重试'.format(num=str(num)))
                else:
                    print('更新指数行情失败!')
                    raise e
        df["amount"] = df["amount"] * 1000; #tushare的交易额单位为(千元)
        df["pct_chg"] = df["pct_chg"] / 100
        return df

    @classmethod
    def renew(cls,conn:sqlite3.Connection):
        cls.get_data('399300.SZ',None,None,conn)
        print('沪深300行情更新完毕！')
    mapper = {'IndexMarketDaily':{'IndexCode':'ts_code','Time':'trade_date','Open':'open','High':'high','Low':'low','Close':'close','Volume':'vol','Amount':'amount'}}
