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

#测试读取包内所有模组;测试获取抽象类所有实现类
# from core.fundamentals.DataCollectionImpls import *
# classes = [cls.__name__ for cls in DataCollection.DataCollection.__subclasses__()]
# print(classes)

#测试get_featurefromDB
# from core.features.getfeature import *
# import datetime
# now = datetime.datetime.now()
# get_featureFromDB(['000001'],'testfe',now-datetime.timedelta(weeks=520),now,'w',{})

# #测试get_feature
# if(__name__=='__main__'):
#     from core.features.DescriptorImpls import *
#     from core.features.getfeature import get_feature
#     import datetime
#     import time
#     t1 = time.time()
    # aa = get_feature(descriptor=Beta(params={'period':100,'index':'399300.SZ'}, freq='d',prepareProc=['winsor']),starttime= datetime.datetime.strptime('20191220','%Y%m%d'),endtime= datetime.datetime.strptime('20200113','%Y%m%d'),check=False)
    # aa = get_feature(CapitalGainOverhang(params={'N':201}, freq='d',prepareProc=['winsor']),datetime.datetime.strptime('20180101','%Y%m%d'), datetime.datetime.strptime('20200113','%Y%m%d'),check=True)
    # aa = get_feature(descriptor=Beta(params={'period':100,'index':'399300.SZ'}, freq='d',prepareProc=['winsor']),starttime= datetime.datetime.strptime('20180101','%Y%m%d'),endtime= datetime.datetime.strptime('20200113','%Y%m%d'),check=True)
    # aa = get_feature(descriptor=Size(freq='d',prepareProc=['winsor','ln','std'],params={'comment':'logtest'}),starttime= datetime.datetime.strptime('20180101','%Y%m%d'),endtime= datetime.datetime.strptime('20200113','%Y%m%d'),check=True)
    # aa = get_feature(descriptor=Illiq(freq='d', prepareProc=None, params={'N': 90}),starttime= datetime.datetime.strptime('20180101','%Y%m%d'),endtime= datetime.datetime.strptime('20200113','%Y%m%d'),check=True)
    # aa = get_feature(descriptor=BTOP(freq='d', prepareProc=None),starttime= datetime.datetime.strptime('20180101','%Y%m%d'),endtime= datetime.datetime.strptime('20200113','%Y%m%d'),check=True)
    # aa = get_feature(descriptor=PastReturn(params={'wait':0,'cum':5},freq='d'),starttime= datetime.datetime.strptime('20180101','%Y%m%d'),endtime= datetime.datetime.strptime('20200113','%Y%m%d'),check=True)
    # print(time.time()-t1)

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
if(__name__=='__main__'):
    from core.model.MultiFactor import MultiFactor
    from datetime import datetime
    from core.features.DescriptorImpls import *
    oldModel = MultiFactor.load_model()
    Freturn = oldModel.Freturns
    forcast = oldModel.forcastStkReturn

    print(bb)


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
#     ts.set_token('46be315f285d45d0db65eb8cba19b61a6b90599169230ed2b83d2470')
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