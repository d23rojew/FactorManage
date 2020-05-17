from core.fundamentals.DataCollectionImpls import *
import sqlite3

#更新交易日历
def renewTradingDays(conn:sqlite3.Connection):
    TushareImpls.CollectTradingDay.renew(conn)

#直接调用所有DataCollection实现类的renew方法;各实现类自定义如何检测缺漏数据,以及更新时间
def renewall(conn:sqlite3.Connection):
    #在这里执行所有DataCollectionImpls实现类的renew方法.暂时只执行TushareImpls.renew
    for cls in DataCollection.DataCollection.__subclasses__():
        cls.renew(conn=conn)
