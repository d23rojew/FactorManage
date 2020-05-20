'''
author: dongruoyu
time: 2020-01-17
1、根据DBdefinition.xml创建或修改数据库表结构;
2、数据自检:检查DBdefinition所定义表中的数据是否最新,否则自动调用DataCollectionImpl更新(只提供更新建议，需人工确认是否更新)
'''
import sqlite3
import xml.etree.ElementTree as ET
import time

#重建表结构
def reconstruct(DBdefinitionXML:ET.ElementTree,conn:sqlite3.Connection):
    #由DBdefinition重建sqlite表结构:若无同名表则新建，若有同名表->删除多余列,增加空缺列
    root = DBdefinitionXML.getroot()
    tableNode = root.findall("./tables//table")
    for table in tableNode:
        tableName = table.attrib['name']
        primaryKeys = table.findall("./property[@identifier='true']")
        allColumns = table.findall("./property")
        sqlGetPK = "select name from pragma_table_info('{tablename}') where pk <> 0".format(tablename=tableName)
        cr = conn.cursor()
        cr.execute(sqlGetPK)
        PKinXML_set = {node.attrib['name'] for node in primaryKeys}
        PKinSQLite_set = {tuple[0] for tuple in cr.fetchall()}
        sqlGetAll = "select name from pragma_table_info('{tablename}')".format(tablename=tableName)
        cr.execute(sqlGetAll)
        AllColinXML_set = {node.attrib['name'] for node in allColumns}
        AllColinSQLite_set = {tuple[0] for tuple in cr.fetchall()}
        colSQL, pkSQL = '', ''
        for node in allColumns:
            if (node in primaryKeys):
                pkSQL += node.attrib['name'] + ','
            colSQL += node.attrib['name'] +' ' + node.attrib['type'] +','
        colSQL = colSQL.strip(',')
        pkSQL = pkSQL.strip(',')
        if(PKinSQLite_set!=PKinXML_set):
            #XML定义主键与数据库现存表主键不同或数据库不存在该表时,按XML定义重新建表
            msg = "表{tablename}设计主键{PKinXML}与现存主键{PKinSQLite}不同，即将开始重建!".format(tablename=tableName,PKinXML=str(PKinXML_set),PKinSQLite=str(PKinSQLite_set)) if PKinSQLite_set!=set() \
                else "数据库中无表{tablename},即将开始创建!".format(tablename=tableName)
            print(msg)
            reconSQL = '''drop table if exists {tablename};
            create table {tablename} ({columns} ,primary key({primarykey}))
            '''.format(tablename=tableName,columns=colSQL,primarykey=pkSQL)
            cr.executescript(reconSQL)
            conn.commit()
        elif(PKinSQLite_set==PKinXML_set and AllColinXML_set!=AllColinSQLite_set):
            #若主键未改变且字段有改变,则将数据库表列根据XML定义进行修改
            interSectCols = ''
            for col in AllColinXML_set&AllColinSQLite_set:
                interSectCols+=col+','
            interSectCols = interSectCols.strip(',')
            alterSQL = '''drop table if exists {tablename}_temp;
            create table {tablename}_temp ({columns} ,primary key({primarykey}));
            Insert into {tablename}_temp({interSection}) select {interSection} from {tablename};
            drop table {tablename};
            alter table {tablename}_temp rename to {tablename}
            '''.format(tablename=tableName,columns=colSQL,primarykey=pkSQL,interSection=interSectCols)
            duoyu_str = str(AllColinSQLite_set-AllColinXML_set) if AllColinSQLite_set-AllColinXML_set!=set() else "{}"
            queshi_str = str(AllColinXML_set-AllColinSQLite_set) if AllColinXML_set-AllColinSQLite_set != set() else "{}"
            msg = "表{tablename}多余字段{duoyu},缺失字段{queshi},将按照xml定义修改表结构".format(tablename=tableName,duoyu=duoyu_str,queshi=queshi_str)
            print(msg)
            t1 = time.time()
            cr.executescript(alterSQL)
            t2 = time.time()-t1
            print("改表花了{sec}秒".format(sec=str(t2)))
            conn.commit
        else:
            #若该表字段未变,则直接跳过
            continue
    print("表结构校验完毕!")


if (__name__ == "__main__"):
    Domtree = ET.parse('../resource/DBdefinition.xml')
    conn = sqlite3.connect('testdb')
    reconstruct(Domtree, conn)
    # from core.fundamentals.DataCollectionImpls.UpdateData import *
    # renewTradingDays(conn)
    # renewall(conn)