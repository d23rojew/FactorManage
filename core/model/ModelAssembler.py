'''
输入:现有持仓权重+股价+调仓费率+固定费用
     股票因子暴露、因子预期收益、因子协方差、最新股价
     在此基础上，给出增量收益最大、增量风险最小的调整方案
输出:还是以权重的形式输出
可归因性:
       权重优化器的每次调仓，返回调仓权重、调仓成本、预期收益
'''
import pandas as pd
import math
import numpy as np
from datetime import datetime,timedelta
import tushare as ts
from core.fundamentals.getfundamentals import fundamentalApi as fApi
from core.Connection.Account import Account
import gzip
import pickle
from core.model.MultiFactor import MultiFactor

def portfolioOptimizor(stk_return:pd.Series, stk_price:pd.Series,stk_cov:pd.DataFrame, stk_hold:dict, feeRate:float, method:str='greedy',gamma:float=0.1):
    '''
    根据当前账户总量、持股比例、股票信息【现价、因子暴露】、因子信息【预期收益，协方差矩阵】、费率
    计算最佳调整量
    :param stk_return:股票预期收益【cols:股票预期收益】（不包含现金）
    :param stk_price:股票价格【cols:股票价格price,rows:股票代码】(包含现金)
    :param stk_cov:股票协方差
    :param stk_hold:股票持仓量【values:持仓股数,keys:股票代码】（包含现金）
    :param feeRate:交易费率(买入+卖出)
    :param method: markowitz/plain
                  ->markowitz:最大化风险调整后的组合收益；
                  ->greedy:以如下顺序迭代优化:
                            存在两个列表:
                            股票置入收益表[置入收益(=股票预期收益-买入/卖出双边手续费率;现金的置入收益为0)
                                         |余额(初始1/30,现金为无穷)]
                                         (置入收益高->低排序)
                            持有列表[置出收益(=-股票预期收益-买入/卖出双边手续费率;现金的置出收益为0
                                             |余额(初始为本日可售卖量，现金为资金余额)]
                                             (置出收益高->低排序)
                            形成边际置换收益匹配表-->（像买卖订单一样，将两表置入收益+置出收益>0的项匹配起来）
                                最高置入收益(余额100元) + 最高置出收益(余额80元) > 0 ->成功置换(80元)
                                最高置入收益(余额20元) + 次高置出收益(余额30元) > 0 ->成功置换(20元)
                                ...
                                直到置入收益 + 置出收益 <= 0元 或一方列表数量为0(基本为持有列表一方)
                            形成调整单->置入股票
                            旧持仓向量为w_old。
                            所有资产持仓限额为总资产1/30。
                            从预期收益最低的持有资产开始，loop:
                                确定仍有额度的股票资产中收益最高的股票
                                若1.替换额度【min(该股票持有数,待置入股票剩余额度,现金余额-(双边替换的金额差)补仓差额）】>0
                                  2.将资产中预期收益最高的资产(计双资产手续费)or替换为现金(计单资产手续费)是否可带来增量收益;
                                计算该被替换资产是否
    :return:orderbook调仓订单(无cash)、调仓量/持有量/调后权重(有cash)、调后组合预期收益、调后组合预期风险
    '''
    stk_hold = pd.Series(stk_hold,name='remain_quantity')
    stk_price.name = 'stk_price'
    stk_return.name = 'stk_return'
    #持仓表
    position_list = pd.DataFrame().merge(stk_hold,how="outer",left_index=True,right_index=True)\
                            .merge(stk_price,how="left",left_index=True,right_index=True)\
                            .merge(stk_return,how="left",left_index=True,right_index=True)      #可调整余额初始为持仓股数
    position_list.sort_values(by='stk_return',ascending=True,na_position='last',inplace=True)
    position_list["amount"] = position_list["stk_price"] * position_list["remain_quantity"]
    #账户总净值
    totalNav = pd.Series.sum(position_list['remain_quantity']*position_list['stk_price'])
    orderbook = {}  # 该订单只要求交易引擎依次尽力执行(出现非整手交易or金额不足时尽力接近订单要求)
    if(method=='greedy'):
    #-----------------------贪心算法-------------------------------
        #形成置出额度排列表
        stkhold_list =  position_list.copy(deep=True)
        stkhold_list.drop(labels=['cash'],inplace=True)#账户总净值
        cash_remain = stk_hold['cash']  #剩余现金
        #形成置入额度排列表
        buy_list = pd.DataFrame().merge(stk_return,how="outer",left_index=True,right_index=True)\
                                 .merge(stk_price,how="left",left_index=True,right_index=True)\
                                 .merge(stk_hold,how="left",left_index=True,right_index=True)\
                                 .sort_values(by='stk_return',ascending=False,na_position='last')
        buy_list['remain_quantity'].fillna(value=0,inplace=True)
        holdlimit_1div30 = totalNav/30/buy_list['stk_price']
        hold_limit = holdlimit_1div30 - buy_list['remain_quantity']
        buy_list['remain_quantity'] = hold_limit.clip(lower=0)
        buy_list.drop(buy_list[buy_list['remain_quantity']<=0].index,inplace=True)
        buy_list["amount"] = buy_list["stk_price"] * buy_list["remain_quantity"]
        #从收益最低的持仓股票、收益最高的可加仓股票开始成对置换，直至换仓无法带来正预期收益
        def swapRet()->tuple:
            #持仓股置换为现金(清仓)的边际收益
            sellRet = -np.inf if stkhold_list.__len__()==0 else -stkhold_list.iloc[0]['stk_return'] - feeRate
            #现金购入股票的边际收益
            buyRet = -np.inf if buy_list.__len__()==0 else buy_list.iloc[0]['stk_return'] - feeRate
            #持仓股置换成高收益股票的边际收益
            stockswapRet = buyRet + sellRet
            return (stockswapRet,buyRet*(cash_remain>0),sellRet)
        while np.nanmax(swapRet())>0:
            stockswapRet,buyRet,sellRet = swapRet()
            maxRet = np.nanmax((stockswapRet,buyRet,sellRet))
            code_buy = None if buy_list.__len__()==0 else buy_list.index[0]
            price_buy = None if buy_list.__len__()==0 else buy_list.loc[code_buy,"stk_price"]
            code_sell = None if stkhold_list.__len__()==0 else stkhold_list.index[0]
            price_sell = None if stkhold_list.__len__()==0  else stkhold_list.loc[code_sell,"stk_price"]
            if(maxRet == buyRet):
                #将现金换为持仓股
                quantity_buy = min(buy_list.loc[code_buy,'remain_quantity'],cash_remain/price_buy)
                quantity_sell = 0
            elif(maxRet == stockswapRet):
                #将持仓股置换为其它股
                amount = min(buy_list.loc[code_buy,'amount'],stkhold_list.loc[code_sell, 'amount'])
                quantity_buy = amount / price_buy
                quantity_sell = amount / price_sell
            elif(maxRet == sellRet):
                #将持仓股置换为现金
                quantity_buy = 0
                quantity_sell = stkhold_list.loc[code_sell,'remain_quantity']

            for code, quantity, status_list, sign, price in ((code_buy, quantity_buy, buy_list, 1,price_buy)
                                                      ,(code_sell, quantity_sell, stkhold_list, -1,price_sell)):
                # 记入订单字典
                if quantity>0:
                    if not orderbook.__contains__(code):
                        orderbook[code] = sign * quantity
                    else:
                        orderbook[code] += sign * quantity
                    # 修改状态列表
                    status_list.loc[code,"remain_quantity"] -= quantity
                    status_list.loc[code,"amount"] = status_list.loc[code,"remain_quantity"] * status_list.loc[code,"stk_price"]
                    if status_list.loc[code,"remain_quantity"] <= 0:
                        status_list.drop(labels=code,inplace=True)
                    cash_remain -= quantity * price * sign
                    if(orderbook.__contains__('cash')):
                        orderbook['cash'] -= quantity * price * sign
                    else:
                        orderbook['cash'] = -quantity * price * sign
    #-------------------贪心算法end-----------------------------------
    #其它算法...
    #-----------------------------------------------------------------
    #根据已生成订单簿，给出:调整后账户画像:调整后组合权重、调整预期增量风险
    after_adj_hold = pd.DataFrame().merge(stk_hold, how="outer", left_index=True, right_index=True) \
                   .merge(pd.Series(data=orderbook,name='trade_shares'),how="outer", left_index=True, right_index=True)\
                   .merge(stk_return, how="left", left_index=True, right_index=True)
    after_adj_hold.fillna(value=0,inplace=True)
    after_adj_hold['hold_quantity'] = after_adj_hold['remain_quantity'] + after_adj_hold['trade_shares']
    after_adj_hold = after_adj_hold.merge(stk_price,how='left', left_index=True, right_index=True)
    after_adj_hold['hold_amount'] = after_adj_hold['hold_quantity'] * after_adj_hold['stk_price']
    after_adj_hold['weight'] = after_adj_hold['hold_amount'] / totalNav
    expRet = sum(after_adj_hold['weight'] * after_adj_hold['stk_return'])
    expStockRet = after_adj_hold['stk_return']
    if(stk_cov is not None):
        cov_merge = stk_cov.merge(after_adj_hold['weight'],how='left',left_index=True,right_index=True)
        expVar = np.matmul(np.matmul(cov_merge['weight'].values,stk_cov.values),cov_merge['weight'].values)
    else:
        expVar = np.nan
    if(orderbook.keys().__contains__('cash')):
        orderbook.pop('cash')
    return {'orderbook':orderbook,'after_adj_hold':after_adj_hold,'expRet':expRet,'expVar':expVar,'expStockRet':expStockRet}

class DecisionMaker:
    #每日末，获取真实持仓数据
    #查看日历、模型表、目标表,根据是否回仓确认优化基础
    #查看日历、模型表，确定需激活的模型,顺次优化组合、计算目标，计入目标表
    #次日开盘时针对目标表下单；针对回仓记录，记收益归属

    MODEL_BEGIN_DATE = '20100101'

    def __init__(self,account:Account):
        self.account:Account = account

    #优化持仓的基底持仓组合；形如字典{stockcode:amount},'cash'代表现金
    base = {};

    @classmethod
    def dividendHandler(cls,baseHold:dict,baseDate:datetime,nowDate:datetime):
        '''
        比对baseDate与nowDate两个日期之间的送股转增事件，将baseDate的持股数量
        转换为nowDate的持股数量
        '''
        return baseHold

    @classmethod
    def evaluatePortfolio(cls,portfolio:dict)->float:
        '''
        todo
        根据当前市场价格评估组合净值
        :param portfolio: {stockcode:holdnumber}
        :return: 总资产
        '''
        return 1000000

    @classmethod
    def getActiveModelStackFromDB(cls)->list:
        '''
        返回当日激活[即到了换仓周期]的模型列表(按模型预测期降序排列)
        :return:
        '''
        CalendarCount = fApi.trade_cal('SSE', cls.MODEL_BEGIN_DATE, datetime.now().strftime('%Y%m%d'), '1').__len__()
        sql = 'select Id,Model,Optimizor,ForcastPeriod from MultiFactorOptimizor where {CalendarCount}%ForcastPeriod=0 and IsInService=1 order by ForcastPeriod desc'
        cursor = fApi.conn.execute(sql.format(CalendarCount=CalendarCount))
        rst = cursor.fetchall()
        unziped = [(x[0],pickle.loads(gzip.decompress(x[1])),x[2],x[3]) for x in rst]
        return unziped


    def getTopModelFromDB(self)->dict:
        #从数据库获取未清算模型中的最短周期模型。
        #日终决策和日初执行都会从表OrderBook中获取未清算顶层模型。
        #日终决策,在rebase标记关闭仓位后，获取剩下的未清算顶层模型作为优化基础(作为base进入本对象域)；
        #日初执行时获取未清算顶层模型目标作为交易目标(直接返回)
        conn = fApi.conn
        sql = 'select b.Model model from OrderBook a left join MultiFactorOptimizor b on a.Id=b.Id ' \
              'where a.IsClosed=0 order by b.ForcastPeriod desc'
        rstCursor = conn.execute(sql)
        tup = rstCursor.fetchall()
        TodayActiveModelList = [pickle.loads(gzip.decompress(x[0])) for x in tup] #里面是MultiFactor对象
        myNav = self.account.getNav()
        #如果没有任何未关闭仓位，则以全现金作为优化基底
        if TodayActiveModelList.__len__() ==0 :
            self.base = {'cash':myNav}
            return {}
        else:
            topModel:MultiFactor = TodayActiveModelList.pop()
            basePeriod = topModel.forcast_period
            sql = 'select a.stockcode,a.post_position amount,a.post_position_plan target, a.AdjDt from OrderBookDetail a ' \
                  'left join OrderBook b ' \
                  'on a.Id=b.Id and a.AdjDt=b.AdjDt ' \
                  'left join MultiFactorOptimizor c on a.Id=c.Id ' \
                  'where b.IsClosed=0 and c.ForcastPeriod=?'
            baseHold = pd.read_sql_query(sql,fApi.conn,params=(basePeriod,),index_col='stockcode')
            pastBaseHold = baseHold['amount'].to_dict()
            baseDate = datetime.strptime(baseHold['AdjDt'][0],'%Y%M%d')
            #将过去的持仓基底根据送赠股情况转化为现在的基底,并用现金项填补子模型活动带来的净值变化
            nowBaseHold = self.dividendHandler(pastBaseHold,baseDate,datetime.now())
            cashDiff = myNav - self.evaluatePortfolio(nowBaseHold)
            nowBaseHold['cash'] = nowBaseHold['cash'] + cashDiff
            self.base =  nowBaseHold
            #直接返回顶层模型的目标
            return {'targetDict':baseHold['target'].to_dict(),'topModelPeriod':basePeriod}

    #region 日终部分(决策模块)
    def reBase(self):
        #从表OrderBook和表Dividend中确定今日哪些订单关闭(在OrderBook中标记关闭行IsClosed=1)
        #随后将新的base载入本类中
        conn = fApi.conn
        CalendarCount = fApi.trade_cal('SSE',self.MODEL_BEGIN_DATE,datetime.now().strftime('%Y%m%d'),'1').__len__()
        sql = 'replace into OrderBook ' \
              '(Id,AdjDt,exp_ret,real_ret,IsClosed) ' \
              'select a.Id,a.AdjDt,a.exp_ret,a.real_ret,1 from OrderBook a left join MultiFactorOptimizor b ' \
              'on a.Id = b.Id and a.IsClosed=0 and {CalendarCount}%b.ForcastPeriod=0'.format(CalendarCount=CalendarCount)
        conn.execute(sql)
        conn.commit()
        self.getTopModelFromDB()

    def makeOptimizeOrder(self):
        #从表MultiFactorOptimizor中获取今日激活运行的优化器，从基层到
        #高层(短周期)逐个执行；
        #每执行完一层改变一层当前对象base，并将新target插入OrderBook表；
        #执行前检查base周期是否可整除当前模型周期；若不可整除则跳过该层模型。
        #每日第二个执行
        rst = self.getActiveModelStackFromDB()
        num_of_models = rst.__len__()
        if(num_of_models==0):
            #本日非调仓日，无需操作
            print('本日无模型调仓!')
            return
        else:
            print('本日以下'+num_of_models.__str__()+'个模型调仓：')
            [print('{id}号模型，预测期{p}日'.format(id=str(id),p=str(forcastperiod))) for id,model,optmizor,forcastperiod in rst]
        stk_price = fApi.quotes(asset_code=None,starttime=None,endtime=datetime.now(), period=1, fields='Close', freq='day', adj=None)
        #todo 这里是以收盘价格近似次日开盘价格，若当日为除权登记日(即这两个价格会有较大差异时)时不可直接近似，需要对stk_price做除权处理
        stk_price.set_index('StockCode',inplace=True)
        cashrow = pd.DataFrame({'Close':1},index=['cash'])
        stk_price = stk_price.append(cashrow)
        stk_price = stk_price['Close']
        for id,model,optmizor,forcastperiod in rst:
            model:MultiFactor
            stk_return = model.forcastStkReturn()['e_ret']
            #multifactormodel.forcastStkCov需另写
            OptimizeOutcome = portfolioOptimizor(stk_return = stk_return
                                   ,stk_price = stk_price
                                   ,stk_cov=model.forcastStkCov()
                                   ,stk_hold=self.base
                                   ,feeRate=0.003
                                   ,method=optmizor
                                   ,gamma=0.1)
            target = OptimizeOutcome['after_adj_hold']['hold_quantity'].to_dict() #获取调整后持仓目标
            exp_ret = OptimizeOutcome['expRet']
            self.base = target
            #优化后，把新目标写入self.base、数据库OrderBook表、OrderBookDetail表
            sql = 'insert into OrderBook(Id,AdjDt,exp_ret,IsClosed) values(?,?,?,?)'
            fApi.conn.execute(sql,(id,datetime.now().strftime('%Y%m%d'),exp_ret,0))
            #再将目标插入OrderBookDetail表(内含现金行)
            for idx,row in OptimizeOutcome['after_adj_hold'].iterrows():
                stockcode = idx
                fore_position = row['remain_quantity']
                adj_amount_plan = row['trade_shares']
                post_position_plan = row['hold_quantity']
                stk_exp_ret = row['stk_return']
                stk_sql = 'insert into OrderBookDetail(Id,AdjDt,stockcode,fore_position,adj_amount_plan,post_position_plan,exp_ret)' \
                          'values (?,?,?,?,?,?,?)'
                fApi.conn.execute(stk_sql,(id,datetime.now().strftime('%Y%m%d'),stockcode,fore_position,adj_amount_plan,post_position_plan, stk_exp_ret))
            fApi.conn.commit()
    #endregion

    # region 日初部分(订单执行与结算)
    def fire(self):
        #次日开盘时运行，调用OrderCount获取下单调整量，逐行向account中发送指令
        #Part1 计算下单量...(考虑现金可买入，且要求整手)
        #获取调整目标
        topModelInfo = self.getTopModelFromDB()
        target_dict = topModelInfo.get('targetDict')
        topModelPeriod = topModelInfo.get('topModelPeriod')
        #获取真实持仓
        real_position = self.account.getPosition()
        real_positionDict = {dic['证券代码']:dic['当前持仓'] for dic in real_position}
        #比对目标与实际差异，获取订单数量
        stocks = set(target_dict.keys()).union(set(real_positionDict.keys()))
        orderAmountDict = {}#{stkcode:(orderAmt,targetAmt)}
        for i in stocks:
            if(i!='cash'):
                targetAmount = 0 if target_dict.get(i) is None else target_dict.get(i)
                realAmount = real_positionDict.get(i)
                orderAmount = int((targetAmount - realAmount)/100)*100
                orderAmountDict[i] = (orderAmount,targetAmount)
        orders = orderAmountDict
        #Part 2 执行orders...
        for k,v in orders:
            orderAmt,targetAmt = v
            if(orderAmt>0):
                self.account.placeOrder('buy', k, orderAmt)
            elif(orderAmt<0):
                self.account.placeOrder('sell', k, orderAmt)
            price = float(ts.get_realtime_quotes(k).iloc[0]['price'])
            holdamt = self.account.getStockPosition(k)
            #Part 3 记录下达效果...
            self.__settleOrder(k,price,holdamt,topModelPeriod)

    def __settleOrder(self,stockcode:str,dealprice:float,holdAmt:int,topModelPeriod:int):
        #被fire调用时执行
        #以股票交易价格作为已结模型的关闭仓位价格、新开模型的开仓价格，每只股票关闭时，尝试对OrderBook表、OrderBookDetail表登记实现损益
        #(这玩意不影响交易流程，但运行的时机和实际交易时机一致)

        #1.标记平仓行实际收益率real_ret
        #1.1 select出OrderBookDetail中所有待平仓行(所有已执行(即标记了开仓价及开仓后仓位)，但未平仓，且指令源模型已到平仓时间的指令)
        #1.2 逐行循环，找出复权因子
        #1.3 计算实现平仓复权价与实现开仓复权价之间的收益率
        #1.4 记录收益率
        CalendarCount = fApi.trade_cal('SSE', self.MODEL_BEGIN_DATE, datetime.now().strftime('%Y%m%d'), '1').__len__()
        markCloseSql = 'select a.Id,a.AdjDt,a.stockcode,a.open_price from OrderBookDetail a ' \
                       'left join MultiFactorOptimizor b on a.Id=b.Id ' \
                       'where a.post_position is not null and a.real_ret is null' \
                       'and {CalendarCount}%b.ForcastPeriod=0 '.format(CalendarCount=CalendarCount)
        closeList = pd.read_sql_query(markCloseSql,fApi.conn)
        for k,v in closeList.iterrows():
            id,adjdt,stkcode,open_price = v
            openAdjFactor = fApi.quotes(stkcode,None,datetime.strptime(adjdt,'%Y%m%d'),1,'Adjfactor','d',None)['Adjfactor'].values[0]
            nowAdjFactor = fApi.quotes(stkcode,None,datetime.now()-timedelta(days=1),1,'Adjfactor','d',None)['Adjfactor'].values[0]
            openAdjPrice = open_price * openAdjFactor
            closeAdjPrice = dealprice * nowAdjFactor
            realRet = closeAdjPrice/openAdjPrice -1
            sql = 'replace into OrderBookDetail' \
              '(id, adjdt, stockcode, fore_position, adj_amount_plan, post_position_plan, post_position, open_price, exp_ret, real_ret)' \
              'select a.id, a.adjdt, a.stockcode, a.fore_position, a.adj_amount_plan, a.post_position_plan' \
              ', a.post_position, {dealprice}, a.exp_ret, {realRet}' \
              'from OrdeBookDetail a' \
              'where a.stockcode=\'{stockcode}\' and a.Id={Id} and a.AdjDt=\'{adjdt}\''.format(dealprice=dealprice,realRet=realRet,stockcode=stockcode,Id=id,adjdt=adjdt)
            fApi.conn.execute(sql)
        #2.标记开仓行的实际开仓价格和仓位(当前阶段，将现实仓位与计划仓位差异全归于顶层模型)
        #找出已下达但未执行(即未标记开仓价格和开仓后仓位）的指令，写入开仓价及开仓后仓位
        markOpenSql = 'replace into OrderBookDetail' \
              '(id, adjdt, stockcode, fore_position, adj_amount_plan, post_position_plan, post_position, open_price, exp_ret, real_ret)' \
              'select a.id, a.adjdt, a.stockcode, a.fore_position, a.adj_amount_plan, a.post_position_plan' \
              ', case when c.ForcastPeriod={topModelPeriod} then {holdAmt} else a.post_position_plan end' \
              ', {dealprice}, a.exp_ret, a.real_ret' \
              'from OrdeBookDetail a' \
              'left join OrderBook b on a.Id=b.Id and a.AdjDt=b.AdjDt' \
              'left join MultiFactorOptimozor c on a.Id=c.Id'\
              'where a.stockcode=\'{stockcode}\' and b.IsClosed=0 and a.post_position is null'.format(topModelPeriod=topModelPeriod,holdAmt=holdAmt,dealprice=dealprice,stockcode=stockcode)
        fApi.conn.execute(markOpenSql)
        fApi.conn.commit()
    #endregion
