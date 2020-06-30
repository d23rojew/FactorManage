from datetime import datetime
import pandas as pd
from typing import Union,List
import sqlite3
import itertools
import multiprocessing
import numpy as np
import time
from .util import getFirstDate
from tqdm import tqdm
from core.features.Descriptor import Descriptor
from core.fundamentals.getfundamentals import fundamentalApi as api
import math
from contextlib import closing

def get_feature(descriptor:Union[List[Descriptor],Descriptor],starttime:datetime, endtime:datetime,stock_name:Union[str,List[str]]=None,freq:str='d',check:bool=True)->pd.DataFrame:
    '''
    :param stock_name: 股票代码
    :param descriptor: 特征对象(Descriptor子类的对象)
    :param starttime: 起始时间
    :param endtime: 结束时间
    :param check: 是否检查并自动计算数据库中的空缺
    :param freq: 调取频率('min'/'d'/'w'/'m'/'fix')
    被调用时,检测特征是否支持该频率,若支持则按频率计算返回，若不匹配:若调用频率在支持频率之上(如调用分钟频，支持日频)则计算低频特征，
    再广播至高频,反之则报错。
    '''
    if(type(stock_name) is not str and type(descriptor) not in Descriptor.__subclasses__()):
        raise ("目前不支持同时进行多资产多特征查询，请重试")
    #descriptor = [descriptor] if type(descriptor) in Descriptor.__subclasses__() else descriptor  #get_feature支持多特征查询改造
    freqList = list(descriptor.calcDescriptor.keys())
    try:
        li = freqList.index(freq)
    except Exception as e:
        raise("频率freq目前仅支持分钟(min)、日(d)、周(w)、月(m)、永久(fix)，当前输入"+str(freq))
    realFreq = None
    for i in range(li,freqList.__len__()):
        if(descriptor.calcDescriptor[freqList[i]] is not None):
            realFreq = freqList[i]
            break
    if(realFreq is None):
        raise("特征{descriptor}无所选频率{freq}以下的算法，请剔除该特征或更换频率".format(descriptor=descriptor.descriptorRefName(),freq=freq))
    if(realFreq == freq):
        return RecursiveCalc(descriptor, starttime, endtime, stock_name, check)
    #获取对外输出DataFrame的时间轴
    ind = api.TradingTimePoints('stock', starttime, endtime, freq).index
    if(ind.__len__()==0):
        return pd.DataFrame()
    timeSpan = pd.DataFrame(index = ind,data={})
    timeSpan['Time'] = timeSpan.index
    descriptor.freq = realFreq
    if(realFreq in ('w','M')):
        starttime = getFirstDate(starttime,realFreq)
    fetMat = RecursiveCalc(descriptor, starttime, endtime, stock_name, check)
    if(realFreq=='d'):
        timeSpan['freqMatch'] = list(timeSpan.index.strftime('%Y%m%d'))
        fetMat['freqMatch'] = list(fetMat.index.strftime('%Y%m%d'))
    elif(realFreq=='w'):
        timeSpan['freqMatch'] = list(timeSpan.index.strftime('%Y%W'))
        fetMat['freqMatch'] = list(fetMat.index.strftime('%Y%W'))
    elif(realFreq=='M'):
        timeSpan['freqMatch'] = list(timeSpan.index.strftime('%Y%m'))
        fetMat['freqMatch'] = list(fetMat.index.strftime('%Y%m'))
    elif(realFreq=='fix'):
        timeSpan['freqMatch'] = 1
        fetMat['freqMatch'] = 1
    ans = timeSpan.merge(fetMat, how='left', on='freqMatch')
    ans.index = pd.DatetimeIndex(ans['Time'])
    ans.columns.name = fetMat.columns.name
    ans = ans.drop(columns=['freqMatch', 'Time'])
    return ans

def RecursiveCalc(descriptor:Descriptor, starttime:datetime, endtime:datetime, stock_name:list=None, check:bool=True)->pd.DataFrame:
    '''
    获取或计算所选频率的特征。返回DataFrame，纵轴:不同时间点 横轴:不同资产
    '''
    #IO优化:如果需计算一系列时序相近的因子，则必从数据库中取时序相近的原料因子/基本面数据。必有大部分原材料都是重合的,每次都取属于重复IO.(数据库层面有缓存则不需考虑)
    #统筹优化:每个Calprocess中可获取计算某时点因子所需一级源数据;
    #        get_feture根据因子依赖关系，可获得依赖树(各层级需要哪些具体到时段的材料数据);
    #        将依赖树对数据库存储做查找，可获得缺失数据；根据缺失数据和依赖树获得最优计算方案及材料数据到内存。
    conn = sqlite3.connect('../fundamentals/testdb')
    # 如果数据完整或不进行校验,则直接取出

    if(stock_name is None):
        stock_name = list(api.stockinfo().index)
    RawRst, MissingInfo = get_featureFromDB(stock_name, descriptor, starttime, endtime,conn,check)
    if (MissingInfo.__len__()==0):
        conn.close()
        rst = RawRst
        for preproc in descriptor.prepareProc:
            if(preproc=='ln'):
                rst = np.log(rst)
            if(preproc=='winsor'):
                if(rst.__len__()>0):
                    downlimit = rst.quantile(0.01, axis=1)
                    uplimit =rst.quantile(0.99, axis=1)
                    rst.clip(downlimit, uplimit, axis=0, inplace=True)
            if(preproc=='std'):
                rst = rst.sub(rst.mean(axis=1),axis=0).div(rst.std(axis=1),axis=0)
        return rst
    # 若检验发现不存在，则临时计算,并将结果存储返回
    else:
        if(descriptor.ctrlDescriptorList.__len__()==0):
            multicalprocess(descriptor,MissingInfo,conn)
        else:
            multiRegressprocess(descriptor,MissingInfo,conn)
        conn.close()
        return RecursiveCalc(descriptor, starttime, endtime, stock_name, False)

def get_featureFromDB(stock_name:list,descriptor:Descriptor,starttime:datetime, endtime:datetime,conn:sqlite3.Connection,check:bool=False)->(pd.DataFrame,list):
    '''
    todo: 改造缺失数据判定，当前将null数据也当成不存在，但null可能是基础数据缺失导致特征缺失的结果.可以改为认为只要存在该条数据记录，就认为feature存在。
    todo: missing info 支持不同频率
    给定标的序列、特征名称、参数、时间频率、时间段, 检查数据库，返回完整的数据OR缺失数据的标记(标的, 特征名, 时间段, 频率, 参数)
    '''
    if(type(stock_name) is str):
        stock_name=[stock_name]
    starttime_str = starttime.strftime('%Y%m%d')
    endtime_str = endtime.strftime('%Y%m%d')
    dates = api.Tdate[descriptor.freq]
    dates = dates[(dates>=starttime_str) & (dates<=endtime_str)]
    codes = ""
    for code in stock_name:
        codes += ",'"+code+"'"
    codes = codes.strip(',')
    sql = "select * from StockFeatures where Time between '{starttime}' and '{endtime}' and fename = '{fename}' and freq = '{freq}' and StockCode in ({codes}) and value notnull"\
        .format(starttime=starttime_str,endtime=endtime_str,fename=descriptor.SQLName,freq=descriptor.freq,codes = codes)
    df_longRst = pd.read_sql_query(sql,conn)
    df_rst = df_longRst.pivot(index='Time', columns='StockCode', values='value')
    df_rst.index = pd.DatetimeIndex(df_rst.index)
    ReportMissing = []
    if(check):
        ExpectedInfo = set(itertools.product(stock_name,dates))
        ExistInfo = set()
        for index, row in df_longRst.iterrows():
            ExistInfo.add((row['StockCode'],row['Time']))
        MissingInfo = ExpectedInfo-ExistInfo
        if(MissingInfo != set()):
            print('检验特征({fename},频率:{freq})时发现{num}个缺漏信息.'.format(fename=descriptor.SQLName,freq=descriptor.freq,num=str(MissingInfo.__len__()))) #todo:后续此处可以用logger记载
            ReportMissing = [(descriptor,)+ x for x in MissingInfo]
    return df_rst,ReportMissing

def multicalprocess(descriptor:Descriptor, lackInfo:list,conn:sqlite3.Connection):
    # 并发计算(调用实现了Descriptor对象自行计算), 统一存储(与数据库耦合);
    # 最细粒度计算(单标的, 单时间点);
    cores = multiprocessing.cpu_count()
    bufferSize = 500000
    segments = math.ceil(lackInfo.__len__()/bufferSize)
    for s in range(segments):
        with closing(multiprocessing.Pool(processes=cores,maxtasksperchild=math.ceil(bufferSize/cores/10))) as pool:
            beg = s*bufferSize
            end = (beg + bufferSize) if (beg + bufferSize) < lackInfo.__len__() else lackInfo.__len__()
            segList = lackInfo[beg:end]
            resultList = list(tqdm(pool.imap(singleCalcJob,segList),desc='正在计算第{i}段，共{s}段'.format(i=str(s+1),s=segments),total=segList.__len__()))
            for rst in resultList:
                stock_code, date, value = rst
                sql = "Insert or replace into StockFeatures (stockcode,fename,Time,freq,value) values('{stockcode}','{fename}','{Time}','{freq}',{value})"\
                    .format(stockcode=stock_code, fename=descriptor.SQLName, Time=date, freq = descriptor.freq, value=value)
                try:
                    conn.execute(sql)
                except Exception as e:
                    raise(e)
            t1 = time.time()
            conn.commit()
            t2 = time.time()
            print("入库耗时"+str(t2-t1)+"秒!")
    print('特征计算完毕！')

def singleCalcJob(tuple:tuple):
    descriptor, stock_code, date = tuple
    rst = descriptor.calcDescriptor[descriptor.freq](stock_code, date)
    if(rst is None or (type(rst) is not str and np.isnan(rst))):
        rst= 'null'
    return (stock_code, date, rst)

def multiRegressprocess(lackInfo:list,conn:sqlite3.Connection):
    '''
    对回归残差特征存在缺失的时间点，取粗特征与控制特征在该时间点的值，回归取残差后存入数据库。
    :param lackInfo:缺失数据信息
    :param conn:与sqlite的连接
    '''
    descriptor = lackInfo[0][0]
    rawDescriptor = descriptor.__class__(params=descriptor.params,freq=descriptor.freq,prepareProc=descriptor.prepareProc)
    lackTimes = set([x[2] for x in lackInfo])
    for t_str in tqdm(lackTimes,"正在计算回归特征"+descriptor.__class__.__name__):
        t_datetime = datetime.strptime(t_str,'%Y%m%d')
        regresee = get_feature(descriptor,t_datetime,t_datetime,True)
        regressor = [get_feature(ctrlDescriptor,t_datetime,t_datetime,True) for ctrlDescriptor in descriptor.ctrlList]
        mat = pd.concat([regresee,*regressor])


        #todo 将regresee与regressor按股票对齐并回归，将残差对应到股票中并存储.