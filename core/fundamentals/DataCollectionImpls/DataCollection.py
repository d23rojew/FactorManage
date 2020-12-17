'''
author: dongruoyu
time: 2020-01-10
根据DBdefinition.xml及外部数据采集api更新基础数据库中的信息。
'''
import abc
import pandas as pd
import sqlite3
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import *

class DataCollection(abc.ABC):
    '''
    DataCollection提供了数据采集类的模板，其子类仅需实现apidata、mapper、renew三个方法即可完成
    数据的外部获取、格式转换、落地存储、缺漏查询、数据补全功能。mapper提供了内外部数据格式信息，
    因此apidata\renew必须与数据库结构解耦，不得直接通过sql语句查询or读取数据库内信息。
    '''
    @classmethod
    @abc.abstractmethod
    def apidata(cls, StockCode:Union[str,list], starttime:datetime, endtime:datetime)->pd.DataFrame:
        '''
        输入标的代码、起止时间，输出行情序列DataFrame;该DataFrame应与基础数据库中的目标表的字段
        相对应（至少主键及目标字段应当对应）
        可以在此处试图并发下载，优化原生api调用方式
        '''
        return pd.DataFrame()

    @classmethod
    @abc.abstractmethod
    def mapper(cls):
        '''
        一个字典，将apidata查询结果与基础数据表字段相匹配，形如{'StockMarketData'(目标表名):{'Close'(目标表字段):'close','Volume':'vol',...}}
        要求:外层字典只有一项，即"目标表名":{API->目标字段匹配字典}
        内层字典{API->目标字段匹配字典}可有多项，但1、必须包含目标表的主键;2、API回传字段必须完全覆盖mapper内字段。
        '''
        pass

    #获取表结构
    __Domtree = ET.parse('./MyResource/DBdefinition.xml')
    root = __Domtree.getroot()

    @classmethod
    @abc.abstractmethod
    def renew(cls,conn:sqlite3.Connection,checkAlll:bool=False):
        '''
        开放给外部程序的更新方法,在被调用时自定更新方式(如仅每周更新/手动确认是否更新/etc.)
        1.自检数据判定缺失，从而生成get_data的参数
          (自检模式可在抽象类中写好:行情更新类;固定参数类;(不需要手动参数))
        2.调用get_data、probe，适当切割待更新数据,分批入库;这里不要搞并发(在下载时搞并发)
        '''
        pass

    @classmethod
    def get_data(cls,StockCode:str, starttime:datetime, endtime:datetime, conn:sqlite3.Connection)->None:
        '''
        对照api中获取的数据、mapper中待更新的字段与DBdefinition.xml中表结构，将新数据insert或update到数据库目标表
        '''
        tablename = list(cls.mapper.keys())[0]
        tablenode = cls.root.find("./tables/basicInfo/table[@name='{}']".format(tablename))
        keynodes = cls.root.findall("./tables/basicInfo/table[@name='{}']/property[@identifier='true']".format(tablename))

        #检查表定义完备性
        if(tablenode==None):
            raise Exception('未在 DBdefinition.xml 中找到表 {} 的定义!'.format(tablename))
        if(keynodes.__len__()==0):
            raise Exception('表 {} 未定义主键!'.format(tablename))

        #检查mapper定义完备性
        for node in keynodes:
            if(node.attrib['name'] not in cls.mapper[tablename].keys()):
                raise Exception('{}.mapper中没有表 {} 的主键{},无法完成API数据到数据库的映射'.format(cls.__class__.__name__,tablename,node.attrib['name']))

        datas = cls.apidata(StockCode, starttime, endtime)
        if(type(datas) is pd.DataFrame and datas.__len__()>0):
            datas.to_sql(name='temp_table', con=conn, if_exists='replace')
            UpdateSql = cls.__SQLgenerate()
            conn.execute(UpdateSql)
            conn.commit()
        else:
            print("未能在外部api中获取{stock}在{begt}~{endt}区间内的数据,已略过".format(stock=str(StockCode),begt=starttime.strftime("%Y%m%d"),endt=endtime.strftime("%Y%m%d")))

    @classmethod
    def __SQLgenerate(cls)->str:
        '''
        仅供get_data方法调用;
        目标是拼出由temp_table更新目标表指定字段的SQL语句
        '''
        tablename = list(cls.mapper.keys())[0]
        keynodes = cls.root.findall("./tables/basicInfo/table[@name='{}']/property[@identifier='true']".format(tablename))
        allfieldNodes = cls.root.findall("./tables/basicInfo/table[@name='{}']/property".format(tablename))
        sql = '''INSERT OR REPLACE INTO {tablename} ({allfields}) SELECT * FROM(SELECT {insertfields} FROM temp_table LEFT JOIN {tablename} ON {condition})'''
        allfields,insertfields,condition = '','',''
        for fld in allfieldNodes:
            field = fld.attrib['name']
            allfields += ',{field}'.format(field = field)
            if(fld in keynodes):#若该字段为主键
                mapperfield = cls.mapper[tablename][field]
                insertfields += ',ifnull({tablename}.{keyfield},temp_table.{mapperfield}) {keyfield}'.format(tablename=tablename,keyfield=field,mapperfield=mapperfield)
                condition += ' and {tablename}.{keyfield} = temp_table.{mapperfield}'.format(tablename=tablename,keyfield=field,mapperfield=mapperfield)
            elif(field in cls.mapper[tablename].keys()):#若该字段非主键,但需更新
                mapperfield = cls.mapper[tablename][field]
                insertfields += ',temp_table.{mapperfield} {keyfield}'.format(keyfield=field,mapperfield=mapperfield)
            else:#不需要更新的字段
                insertfields += ',{tablename}.{keyfield} {keyfield}'.format(tablename=tablename,keyfield=field)
        allfields = allfields.strip(',')
        insertfields = insertfields.strip(',')
        condition = condition.strip(' and')

        return sql.format(tablename=tablename,allfields=allfields,insertfields=insertfields,condition=condition)

    # 目标表的内容(非主键)字段,用于探测数据点是否存在
    NotKeyFields=None
    @classmethod
    def GET_NotKeyField(cls):
       if(not cls.NotKeyFields):
           tablename = list(cls.mapper.keys())[0]
           keynodes = cls.root.findall("./tables/basicInfo/table[@name='{}']/property[@identifier='true']".format(tablename))
           keyfields = [node.attrib["name"] for node in keynodes]
           for k, v in cls.mapper[tablename].items():
                if (k not in keyfields):
                    cls.NotKeyFields = k
                    return k
       else:
           return cls.NotKeyFields
    @classmethod
    def probe(cls,conn:sqlite3.Connection,checkList:pd.DataFrame=None,**keyargs):
        '''
        探针函数,对参数中给出的主键在数据库中查询该条数据是否存在,被renew用于确定是否存在需要更新的数据。
        使用方式:
        (1)以输入参数为查询条件查找对应的内容字段是否在数据库中存在,返回bool值
            如probe(conn,ts_code='000001.SZ',trade_date='20191010')->True
        (2)输入DataFrame,以其中的每一行为查询条件判断数据库中是否存在对应内容字段，将无对应内容字段的行以新DataFrame返回
                                                (目前若该行的条件对应多条数据，只要一条数据为null则判定该条件对应数据存在缺失.但某些api)
                                                todo:增加选项,若该行的条件对应多条数据，需要这些数据全部为null才返回该行为数据缺失).
        '''
        tablename = list(cls.mapper.keys())[0]
        reverse_mapper = {value: key for key, value in cls.mapper[tablename].items()}
        if(checkList is None):
            sql = "select 1 from {targetTable} where {clause} limit 1"
            clause = " "
            for field in keyargs.keys():
                db_field = reverse_mapper[field]
                clause += "{db_field}='{param}' and ".format(db_field=db_field,param=keyargs[field])
            a1 = time.time()
            clause += "{nonkf} notnull".format(nonkf=cls.GET_NotKeyField())
            probesql = sql.format(targetTable=tablename,clause=clause)
            rst = conn.execute(probesql)
            return True if rst.fetchall().__len__()>0 else False
        else:
            if(type(checkList) is not pd.DataFrame):
                raise("checkList必须为pandas.DataFrame!")
            checkList.to_sql(name='temp_table', con=conn, if_exists='replace',index=False)
            joinCondition = ''
            sql = 'select a.* from temp_table a left join {tablename} b on {joinCondition} where b.{contentField} isnull'
            for col in list(checkList.columns):
                joinCondition += 'a.{df_col} = b.{db_col} and'.format(df_col=col, db_col=reverse_mapper[col])
            joinCondition = joinCondition[0:-4]
            probesql = sql.format(tablename = tablename,joinCondition=joinCondition,contentField=cls.GET_NotKeyField())
            rst = pd.read_sql_query(probesql, conn)
            return rst
