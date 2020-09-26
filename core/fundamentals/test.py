# import pandas as pd
# import datetime
# import sqlite3
# import xml.etree.ElementTree as ET
# from core.fundamentals.DataCollectionImpls.DataCollection import DataCollection
#测试DataCollection是否能更新部分数据
# class DataCollectionImpl(DataCollection):
#     def __apidata(self, StockCode:str, starttime:datetime, endtime:datetime)->pd.DataFrame:
#         return pd.DataFrame([[10,'000011','2019-01-02'],[14,'000002','2019-01-02']],columns=['close_api','stockcode_api','time_api'])
#     mapper = {'StockMarketData':{'Close':'close_api','StockCode':'stockcode_api','Time':'time_api'}}
#
# dctest = DataCollectionImpl()
# conn = sqlite3.connect('testdb')
# dctest.get_data(None,None,None,conn)

# 测试datacheck
# Domtree = ET.parse('../resource/DBdefinition.xml')
# conn = sqlite3.connect('testdb')
# from core.fundamentals.DataCollectionImpls.TushareImpls import *
# CollectHS300Daily.renew(conn)

#测试fundamentalApi
# import datetime
# from time import time
# from core.fundamentals.getfundamentals import fundamentalApi as fapi
# t1 = time()
# df = fapi.quotes(asset_code='000001.SZ', starttime=None, endtime=datetime.datetime.strptime('20200124','%Y%m%d'),
#                                               period=30, fields=['Open'], freq='d',
#                                               adj='hfq')
# returnMat = fapi.getReturn(asset_code=None,starttime=datetime.datetime.strptime('20190203','%Y%m%d'), endtime=datetime.datetime.strptime('20200203','%Y%m%d'),forcast_period=10)
# print(time()-t1)
# a = fapi.TradingTimePoints('stockcode',datetime.datetime.strptime('202003241002','%Y%m%d%H%M'),datetime.datetime.strptime('202003251402','%Y%m%d%H%M'),freq='min')
# b = 1
# df1 = fapi.quoteMinute(asset_code='000001.SZ', starttime=datetime.datetime.strptime('20200724'+' 09:30','%Y%m%d %H:%M'), endtime=datetime.datetime.strptime('20200724'+' 15:00','%Y%m%d %H:%M'),
#                                               fields=['Volume'])
# t2 = time()
# print(t2-t1)
# df2 =  fapi.quoteMinute(asset_code='000002.SZ', starttime=datetime.datetime.strptime('202007150946','%Y%m%d%H%M'), endtime=datetime.datetime.strptime('20200725','%Y%m%d'),
#                                               period=None, fields=['Open'])
# print(time()-t2)
# print(1)
#测试Calprocess->CapitalGainOverhang
# from core.features.DescriptorImpls import PastReturn
# from core.fundamentals.getfundamentals import fundamentalApi as fapi
# from datetime import datetime,timedelta
# dt = datetime.strptime('20191216','%Y%m%d')
# ptobj = PastReturn(params={'wait':1,'cum':2},freq='d')
# bb = ptobj.calcDayDescriptor(stock_code='000001.SZ',timepoint='20191216')
# cc = fapi.getReturn(asset_code='000001.SZ',starttime=dt,endtime=dt+timedelta(days=5),forcast_period=2)
# dd = fapi.quotes(asset_code='000001.SZ',starttime=dt-timedelta(days=5),endtime=dt+timedelta(days=5),period=None,fields=['Open','Close'],freq='d',adj='hfq')
# print(bb)

# 测试Calprocess->Beta
# from core.features.DescriptorImpls import *
# beta = Beta(params={'period':60,'index':'399300.SZ'},freq='d')
# betarst = beta.calcDescriptor('999999.SZ', '20191202')
# print(betarst)

# 测试Calprocess->FormulaAlpha1
# from core.features.DescriptorImpls import *
# fAlpha1 = FormulaAlpha1(params={'t':10},freq='d')
# depart = fAlpha1.calcDescriptor['d']('000001.SZ', '20180113')
# print(depart)

# 测试Calprocess->FormulaAlpha2
# from core.features.DescriptorImpls import *
# fAlpha2 = FormulaAlpha2(params={'t':10},freq='d')
# abnormalV = fAlpha2.calcDescriptor['d']('000001.SZ', '20180113')
# print(abnormalV)

# 测试Calprocess->FormulaAlpha3
# from core.features.DescriptorImpls import *
# fAlpha3 = FormulaAlpha3(params={'t':10},freq='d')
# depart = fAlpha3.calcDescriptor['d']('000001.SZ', '20180113')
# print(depart)

#测试读取包内所有模组;测试获取抽象类所有实现类
# from core.fundamentals.DataCollectionImpls import *
# classes = [cls.__name__ for cls in DataCollection.DataCollection.__subclasses__()]
# print(classes)

#测试get_featurefromDB
# from core.features.getfeature import *
# import datetime
# now = datetime.datetime.now()
# get_featureFromDB(['000001'],'testfe',now-datetime.timedelta(weeks=520),now,'w',{})

#测试get_feature
# if(__name__=='__main__'):
#     from core.features.DescriptorImpls import *
#     from core.features.getfeature import get_feature
#     import datetime
#     import time
#     t1 = time.time()
#     aa = get_feature(descriptor=[Beta(params={'period':100,'index':'399300.SZ'}, freq='d',prepareProc=['winsor'])
#                                 ,CapitalGainOverhang(params={'N':201}, freq='d',prepareProc=['winsor'])
#                                 ,Size(freq='d',prepareProc=['winsor','ln','std'],params={'comment':'logtest'})]
#                      ,starttime= datetime.datetime.strptime('20200113','%Y%m%d'),endtime= datetime.datetime.strptime('20200113','%Y%m%d'),check=False)
#     aa = get_feature(CapitalGainOverhang(params={'N':201}, freq='d',prepareProc=['winsor']),datetime.datetime.strptime('20200101','%Y%m%d'), datetime.datetime.strptime('20200113','%Y%m%d'),check=True)
#     aa = get_feature(descriptor=Beta(params={'period':100,'index':'399300.SZ'}, freq='d',prepareProc=['winsor']),starttime= datetime.datetime.strptime('20180101','%Y%m%d'),endtime= datetime.datetime.strptime('20200113','%Y%m%d'),check=True)
#     aa = get_feature(descriptor=Size(freq='d',prepareProc=['winsor','ln','std'],params={'comment':'logtest'}),starttime= datetime.datetime.strptime('20180101','%Y%m%d'),endtime= datetime.datetime.strptime('20200113','%Y%m%d'),check=True)
#     aa = get_feature(descriptor=Illiq(freq='d', prepareProc=None, params={'N': 90}),starttime= datetime.datetime.strptime('20180101','%Y%m%d'),endtime= datetime.datetime.strptime('20200113','%Y%m%d'),check=True)
#     aa = get_feature(descriptor=BTOP(freq='d', prepareProc=None),starttime= datetime.datetime.strptime('20180101','%Y%m%d'),endtime= datetime.datetime.strptime('20200113','%Y%m%d'),check=True)
#     aa = get_feature(descriptor=PastReturn(params={'wait':0,'cum':5},freq='d'),starttime= datetime.datetime.strptime('20180101','%Y%m%d'),endtime= datetime.datetime.strptime('20200113','%Y%m%d'),check=True)
#     aa = get_feature(descriptor=Size(freq='d',prepareProc=['winsor','ln','std'],params={'comment':'logtest'}),starttime= datetime.datetime.strptime('20180101','%Y%m%d'),endtime= datetime.datetime.strptime('20200113','%Y%m%d'),check=True)
#     print(time.time()-t1)

#测试get_feature
# if(__name__=='__main__'):
#     from core.features.DescriptorImpls import *
#     from core.features.getfeature import get_feature
#     import datetime
#     import time
#     t1 = time.time()
#     aa = get_feature(descriptor=Beta(params={'period':100,'index':'399300.SZ'}, freq='d',prepareProc=['winsor']),starttime= datetime.datetime.strptime('20191220','%Y%m%d'),endtime= datetime.datetime.strptime('20191220','%Y%m%d'),check=False)
#     t2 = time.time()
#     print(t2-t1)
#     bb = get_feature(descriptor=Beta(params={'period':100,'index':'399300.SZ'}, freq='d',prepareProc=['winsor']),starttime= datetime.datetime.strptime('20191221','%Y%m%d'),endtime= datetime.datetime.strptime('20191221','%Y%m%d'),check=False)
#     t3 = time.time()
#     print(t3-t2)
#     cc = get_feature(descriptor=Beta(params={'period':100,'index':'399300.SZ'}, freq='d',prepareProc=['winsor']),starttime= datetime.datetime.strptime('20191222','%Y%m%d'),endtime= datetime.datetime.strptime('20191222','%Y%m%d'),check=False)
#     t4 = time.time()
#     print(t4-t3)
#     dd = get_feature(descriptor=Beta(params={'period':100,'index':'399300.SZ'}, freq='d',prepareProc=['winsor']),starttime= datetime.datetime.strptime('20191223','%Y%m%d'),endtime= datetime.datetime.strptime('20191223','%Y%m%d'),check=False)
#     t5 = time.time()
#     print(t5-t4)
#     ee = get_feature(descriptor=Beta(params={'period':100,'index':'399300.SZ'}, freq='d',prepareProc=['winsor']),starttime= datetime.datetime.strptime('20191224','%Y%m%d'),endtime= datetime.datetime.strptime('20191224','%Y%m%d'),check=False)
#     t6 = time.time()
#     print(t6-t5)
#测试multifactor
# if(__name__=='__main__'):
#     from core.model.MultiFactor import MultiFactor
#     from datetime import datetime
#     from core.features.DescriptorImpls import *
#     f1 = Beta(params={'period':100,'index':'399300.SZ'} ,freq='d',prepareProc=['winsor','std'])
#     f2 = Size(freq='d',prepareProc=['winsor','ln','std'],params={'comment':'logtest'})
#     f3 = BTOP(freq='d', prepareProc=['winsor','std'])
#     f4 = CapitalGainOverhang(params={'N':201}, freq='d',prepareProc=['winsor','std'])
#     f5 = Illiq(freq='d', prepareProc=['winsor','std'], params={'N': 90})
#     f6 = PastReturn(freq='d', prepareProc=['winsor','std'], params={'wait':0,'cum':5})
#     newModel = MultiFactor(factors=[f1,f2,f3,f4,f5,f6],
#                            starttime=datetime.strptime('20191001','%Y%m%d'),
#                            endtime=datetime.strptime('20200113','%Y%m%d'),
#                            forcast_period=1,
#                            controls = [], freq='d')
#     newModel.uptodate(datetime.strptime('20200115','%Y%m%d'))
#     newModel.save_model()
#     fret = newModel.Freturns
#     forcast = newModel.forcastStkReturn
#     print(forcast)

#测试multifactor.forcastStkReturn
# if(__name__=='__main__'):
#     from core.model.MultiFactor import MultiFactor
#     from datetime import datetime
#     from core.features.DescriptorImpls import *
#     oldModel = MultiFactor.load_model()
#     Freturn = oldModel.Freturns
#     forcast = oldModel.forcastStkReturn
#
#     print(bb)


#测试tushare并发
# def beginTs(a):
#     import tushare as ts
#     ts.set_token('46be315f285d45d0db65eb8cba19b61a6b90599169230ed2b83d2470')
#     api = ts.pro_api()
#     print(api)
#
# if(__name__=='__main__'):
#     import multiprocessing
#     p = multiprocessing.Pool(8)
#     p.starmap(beginTs,[[1] for i in range(8)])
#     p.close()
#     p.join()


#测试tushare
# import tushare as ts
# import time
# from core.fundamentals.getfundamentals import fundamentalApi as fapi
# if(__name__=='__main__'):
#     ts.set_token('83a82fadb0bbb803f008b31ce09479e5107f4aba3f28d5df2174c642')
#     api = ts.pro_api()
#     j8 = api.fut_basic(exchange = 'CZCE',fields = 'name,fut_code,trade_time_desc')
#     print(j8)

#内存读测试
# import sqlite3
# import pandas as pd
# from core.fundamentals.getfundamentals import fundamentalApi as fapi
# import time
# from tqdm import tqdm
# conn = sqlite3.connect('testdb')
# dates = fapi.trade_cal('SSE','20180108','20180915','1')
# 1==1
# dateli = list(dates['date'].values)
# l = dateli.__len__()
# t1 = time.time()
# for i in tqdm(range(10)):
#     bb = fapi.quotes(stock_code='000001.SZ',starttime=None, endtime=datetime.datetime.strptime(dateli[i],'%Y%m%d'),period=2,freq=None, fields=['Close'],adj=None)
#     # rst = conn.execute("select Time  ,Close from StockMarketDaily where StockCode='000001.SZ' and Time>='{time}' and Time<='{time}' order by Time desc".format(time=dateli[i]))
#     # rst = conn.execute("select 1")
#     # bb = rst.fetchall()
# t2 = time.time()
# print('循环1用时')
# print(t2-t1)
# for i in tqdm(range(10)):
#     bb = fapi.quotes(stock_code='000002.SZ',starttime=None, endtime=datetime.datetime.strptime(dateli[i],'%Y%m%d'),period=2,freq=None, fields=['Close'],adj=None)
# print('循环2用时')
# print(time.time()-t2)

#测试FMB
# from linearmodels import FamaMacBeth
# import numpy as np
# model = FamaMacBeth(np.array([[1,2],[2,3],[3,4]]),np.array([[np.nan,np.nan],[12,13],[13,14]]))

#多进程测试
# print(__name__)
# print(globals())
# if(__name__=='__main__'):
#     import core.fundamentals.testmodule as tm
#     print(__name__)
#     print(globals())
#     tm.runMul()

#tqdm迭代测试
# from multiprocessing import pool
# from time import sleep
# from tqdm import tqdm
# pbar = tqdm(total=100,desc='测试手动滚动',ncols=80)
# for i in range(100):
#     pbar.update(1)
#     sleep(0.1)
# def singleJob(i):
#     sleep(2)
#     stra = "Job:"+str(i)
#     # print(stra + '\n')
#     return(stra)
#
# if(__name__!='main'):
#     print('subprocess!')

# if(__name__=='__main__'):
#    p = pool.Pool()
#    arglist = list(range(20))
#    mulIter = p.imap(func=singleJob,iterable=arglist)
#    bar = tqdm(mulIter,desc='正在迭代')
#    print(bar.__iter__())
#     print(mulIter)
#     while True:
#         print('wawa');
#         print(mulIter.next())

#总结:
#1、进程池p.imap执行后，即会在后台开启进程执行任务；
#2、p.imap返回的IMapIterator对象有迭代方法,该方法可迭代后台进程已完成的任务
#   的产出结果，并且会在暂时无产出时阻塞(可利用这点做多进程进度条)
#3、问题就在能否获取IMapIterator对象后,动态往进程池中提交任务？

#动态提交获取.

#多进程callback测试
# from multiprocessing.pool import Pool
# from time import *
# def stupidJob(i):
#     sleep(1)
#     print(__name__+' run Job:'+str(i))
# def callbackJob(i):
#     print('callback'+__name__ + ':' + str(i))
# if(__name__=='__main__'):
#     p = Pool()
#     for i in range(10):
#         p.apply_async(stupidJob,args=(i,),callback=callbackJob)
#     p.close()
#     p.join()

#CollectStockMinute测试
# if(__name__=='__main__'):
#     from core.fundamentals.DataCollectionImpls.TushareImpls import CollectStockMinute
#     import sqlite3
#     conn = sqlite3.Connection('testdb')
#     # 更新交易日历
#     CollectStockMinute.renew(conn)


#tushare分钟接口测试
# import tushare as ts
# ts.set_token("83a82fadb0bbb803f008b31ce09479e5107f4aba3f28d5df2174c642")
# ans = ts.pro_bar('000001.SH'
#                                        , start_date='2020-07-22 00:00:00'
#                                        , end_date='2020-07-22 23:59:59'
#                                        , freq='1min')
# print('hello')

#Markowitz优化函数开发
# if(__name__=='__main__'):
#     from core.model.MultiFactor import MultiFactor
#     from datetime import datetime
#     from core.features.DescriptorImpls import *
#     from core.features.getfeature import get_feature
#     f1 = Beta(params={'period':100,'index':'399300.SZ'} ,freq='d',prepareProc=['winsor','std'])
#     f2 = Size(freq='d',prepareProc=['winsor','ln','std'],params={'comment':'logtest'})
#     f3 = BTOP(freq='d', prepareProc=['winsor','std'])
#     f4 = CapitalGainOverhang(params={'N':201}, freq='d',prepareProc=['winsor','std'])
#     f5 = Illiq(freq='d', prepareProc=['winsor','std'], params={'N': 90})
#     f6 = PastReturn(freq='d', prepareProc=['winsor','std'], params={'wait':0,'cum':5})
#     [get_feature(descriptor=f,
#                 starttime=datetime.strptime('20191001', '%Y%m%d'),
#                 endtime=datetime.strptime('20200113', '%Y%m%d'), check=True) for f in [f1,f2,f3,f4,f5,f6]]
#     newModel = MultiFactor(factors=[f1,f2,f3,f4,f5,f6],
#                            starttime=datetime.strptime('20191001','%Y%m%d'),
#                            endtime=datetime.strptime('20200113','%Y%m%d'),
#                            forcast_period=1,
#                            controls = [], freq='d')
#     newModel.uptodate(datetime.strptime('20200115','%Y%m%d'))
#     newModel.save_model()
#     print(1)

#测试ECOS_BB
# import cvxpy as cvx
# x = cvx.Variable(2)
# obj = cvx.Minimize(x[0] + cvx.norm(x, 1))
# constraints = [x >= 2]
# prob = cvx.Problem(obj, constraints)
#
# # Solve with ECOS.
# prob.solve(solver=cvx.ECOS_BB)
# print("optimal value with ECOS_BB:", prob.value)

#测试markowitz组合优化器
from core.features.DescriptorImpls import *
from core.model.Markowitz import *
from core.model.MultiFactor import MultiFactor as mf
from core.features.DescriptorImpls import *
from core.features.getfeature import get_feature
from core.fundamentals.getfundamentals import fundamentalApi as fapi
oldmo = mf.load_model()
print('model loaded')
stk_feature = get_feature(oldmo.factors
                          ,starttime=datetime.strptime('20200113', '%Y%m%d')
                          ,endtime=datetime.strptime('20200113', '%Y%m%d')
                          ,freq = 'd'
                          ,check = False)
print('done load feature')
stk_price = fapi.quotes(asset_code=None
                                  ,starttime=None
                                  ,endtime=datetime.strptime('20200113', '%Y%m%d')
                                  ,period=1
                                  ,fields = 'Close'
                                  ,freq='day'
                                  ,adj = None)
print('done get price')
stk_price.set_index(stk_price['StockCode'],inplace=True)
stk_hold = pd.Series(index = stk_price.index
                     ,data =np.floor(np.random.rand(stk_price.__len__())*1000)+100)
f_return = pd.Series(oldmo.expFreturn)
f_cov = oldmo.FreturnCov
print('done get cov and feature return')
Markowitz(stk_feature=stk_feature
          ,stk_price=stk_price['Close']
          ,stk_hold =stk_hold
          ,f_return = f_return
          ,f_cov = f_cov
          ,feeRate = 0.003
          ,totalNV=100000000)