'''
择时兼容:
至少这个模型可以获得因子收益序列,可对因子收益序列进行后继的择时研究。
在缺失择时信号时，可认为因子预期收益恒为平均收益，以(风险调整后因子预期收益)*股票因子暴露生成选股模型.
在存在因子择时信号时，由择时信号给出因子预期收益，同上对股票打分。
这个东西在日终应当能输出因子收益序列、因子平均收益、个股因子暴露
在日内进行因子暴露/收益实时计算暂时不去考虑！！

交易兼容:
日度:
股票交易模块参考因子收益、因子波动率、因子相关性、因子暴露选出最优组合。
每日终跑一次FMB回归模型，提供因子收益序列(及在没有择时给出因子预期收益时，给出因子平均收益和相关性)
跑一次择时模型 输入因子收益序列->获取因子预期收益
每日终跑一次当日特征，获取个股因子暴露

定义好接口及规则：
1、因子质量的选择规则
    该模块用于输出【在用】因子库、因子收益序列、因子协方差矩阵
   (1)对库中所存的所有选股因子，对最近n期做横截面回归，选择系数最显著的前k个且显著度大于p的因子作为选股因子
   (2)对该(<)k个因子做FMB回归，取得因子收益序列，存储为【在用】因子库、因子收益序列、因子协方差矩阵
2、因子择时模块
    用CART随机森林+时序指数库进行择时研究，保存模型用于每日因子收益预测，保存结果用于评估时序模型质量
    (1)模型训练:输入【在用】因子收益序列，输出【在用】模型并存储
    (2)预测:每日日终，在特征指数更新后根据【在用】因子收益择时模型预测一次次日【在用】因子收益。
3、组合形成模块
    根据股票因子暴露、预期因子收益及协方差矩阵，求解股票池有效前沿和切点投资组合
4、确定日常模块
    每日日程:
        日终：
            执行基础信息、特征、特征指数更新任务；
            执行2.2获取【在用】因子收益
            获取已贮存的【在用】因子协方差矩阵、【在用】因子收益、最新个股日特征
            ->根据组合最优化算法获取明日持仓比例，比对当日持仓比例，获得调仓指令并存储
        日初：
            根据调仓指令下单
    每l日
        在当日基础信息、特征、特征指数更新完后
        执行1，更新【在用】因子库、因子收益序列、因子协方差矩阵；
        执行2.1，更新【在用】因子择时模型
---------------------------------
明确经验构建与自动调优之间的界线
描述子(descriptor)需经验构造;
描述子筛选需自动调优;
描述子的因子收益序列建模：简单自回归模型，需自动调优;
    建模:以最近n日作为窗口期,n日均值作为n+1日因子期望收益，统计n+1 avg(abs(因子实现收益-期望收益))作为预测偏误。
    优化n使得该偏误最小，在此之上记录n、均值和偏误作为因子预期收益和标准差

日内：
暂时不去考虑
'''
from datetime import datetime
from core.fundamentals.getfundamentals import fundamentalApi as fapi
from core.features.DescriptorImpls import *
from core.features.getfeature import get_feature
import xarray as xr
import statsmodels.api as sm
import pickle
import sqlite3
from tqdm import tqdm
from typing import List,Union

class MultiFactor:
    conn = sqlite3.connect('../fundamentals/testdb')
    def __init__(self,factors:List[Descriptor],starttime:datetime,endtime:datetime,forcast_period:int,controls:list,freq:str='d'):
        '''
        :param factors: 参与建模的因子列表，类型:list<Descriptor>
        :param starttime: 建模开始时间
        :param endtime: 建模结束时间
        :param forcast_period: 模型预测期(形如'1d'/'2w')
        :param controls: 控制因子列表，类型:list<Descriptor>，个股在该时间点控制因子值为0则该条数据不参与横截面回归
        '''
        #记录模型参数
        self.timespan = [starttime,endtime]            #模型样本起止时间
        self.factors = factors                         #所选因子
        self.forcast_period = forcast_period           #预测期
        self.controls = controls                       #控制因子
        self.freq = freq                               #时间频率
        self.optimalMAperiod = {x.descriptorRefName():20 for x in factors}   #预测因子收益所用均线周期
        #获得模型日期序列、计算因子收益
        self.dateIndex,self.rst = self.__getfundamental(self.timespan)
        #计算因子预期收益
        self.__factorReturnOptimizer()

    def uptodate(self,endtime:datetime):
        '''
        更新因子收益序列、因子收益序列最优预测移动平均线周期至指定日期
        :param endtime:指定的日期
        '''
        if(endtime<=self.timespan[1]):
            raise Exception("所选更新日期{}需大于当前模型已有日期{}!".format(endtime.strftime('%Y%m%d'),self.timespan[1].strftime('%Y%m%d')))
        print("开始更新模型,最新日期{}->{}".format(self.timespan[1].strftime('%Y%m%d'),endtime.strftime('%Y%m%d')))
        new_dateIndex,new_rst = self.__getfundamental((self.timespan[1],endtime))
        if(new_dateIndex[0] == self.dateIndex[-1]):
            new_dateIndex = new_dateIndex[1:]
            new_rst = new_rst[1:]
        self.dateIndex = self.dateIndex.append(new_dateIndex)
        self.rst = self.rst + new_rst
        self.__factorReturnOptimizer()
        self.timespan[1] = endtime
        print("多因子模型更新完毕!")

    def __getfundamental(self,timespan):
        '''
        计算并返回指定时间段内的交易日、描述子数据、收益数据以及横截面回归结果.
        :param timespan: 指定的时间段
        :return: (out_dateIndex,out_endogs,out_exogs,out_dateIndex)
            out_dateIndex:时间段内的交易时点序列
            out_endogs:时间段内的股票收益数据
            out_exogs：时间段内的描述子数据
            out_rst:时间段内的横截面回归结果
        '''
        ReturnMat = fapi.getReturn(asset_code=None,starttime=timespan[0],endtime=timespan[1],forcast_period=self.forcast_period,freq=self.freq,asset_type='stock')
        #取因子矩阵
        FactorMatDict = {'feat_'+x.descriptorRefName() : get_feature(descriptor=x,starttime=timespan[0],endtime=timespan[1],freq=self.freq,check=False) for x in self.factors}
        ControlMatDict = {'ctrl_'+x.descriptorRefName() : get_feature(descriptor=x,starttime=timespan[0],endtime=timespan[1],freq=self.freq,check=False) for x in self.controls}
        #FMB回归
        matchDict = {'return':ReturnMat}
        matchDict.update(FactorMatDict)
        matchDict.update(ControlMatDict)
        matchDs = xr.Dataset(matchDict)
        def ProdTrue(*args):
            for obj in args:
                if(args is True):
                    continue
                else:
                    return False
            return True
        CalibratedCtrls = [matchDs[fe_name].values for fe_name in matchDs if fe_name[0:4]=='ctrl']
        CalibratedFeats = [matchDs[fe_name].values for fe_name in matchDs if fe_name[0:4]=='feat']
        if(CalibratedCtrls.__len__()>0):
            ArrayProdTrue = np.frompyfunc(ProdTrue,CalibratedCtrls.__len__(),1)
            AssembledControlMat = ArrayProdTrue(*CalibratedCtrls).astype('bool')
            MaskedReturn = np.ma.MaskedArray(matchDs['return'].values,mask=~AssembledControlMat,fill_value=np.nan)
        else:
            MaskedReturn = matchDs['return'].values
        l = MaskedReturn.shape[0]
        out_rst = []
        exogs = np.stack(CalibratedFeats)
        noneCount = 0
        for i in tqdm(range(l),desc='正在截面回归...'):
            try:
                out_rst.append(sm.OLS(endog=MaskedReturn[i], exog=exogs[:, i, :].T, missing='drop').fit())
            except:
                out_rst.append(None)
                noneCount+=1
        if(noneCount==l):
            raise Exception("所选时间区间内无有效回归样本，请检查基础数据！")
        out_dateIndex = matchDs['return'].indexes['Time']
        return (out_dateIndex,out_rst)

    def __factorReturnOptimizer(self):
        '''
        使用移动平均线预测因子收益。针对每个因子收益序列，(在上个最优周期周围)找到预测收益离差最小的移动平均线,并给出下期的因子收益预测。
        :param timespan:
        :return:
        '''
        print('开始寻找预测因子收益最优均线期数!')
        upperlim = round(self.rst.__len__() / 2)
        lowlim = 10
        def cutEdge(x):
            x = lowlim if x<lowlim else x
            x = upperlim if x>upperlim else x
            return x
        for f in self.factors:
            originOptimalP = self.optimalMAperiod[f.descriptorRefName()]
            candidateSet = set([cutEdge(round(originOptimalP*x)) for x in (0.6,0.8,1,1.2,1.4)])
            if(candidateSet.__len__()==1):
                self.optimalMAperiod[f.descriptorRefName()] = candidateSet.pop()
            else:
                minDiff = np.inf
                factorName = f.descriptorRefName()
                factorReturnSeries = self.Freturns[factorName]
                for p in candidateSet:
                    newDiff = abs(factorReturnSeries - factorReturnSeries.rolling(p).mean().shift(1)).mean()
                    if newDiff < minDiff:
                        minDiff = newDiff
                        self.optimalMAperiod[f.descriptorRefName()] = p
        print('搜索完毕，最优均线期数为:'+str(self.optimalMAperiod))


    '''
    报告项:
    单因子相关: 因子收益序列(已控制其它因子的)
    因子全貌: 因子暴露相关性矩阵、因子收益协方差矩阵、因子收益序列均值(t值)、因子IC/IR均值
    模型质量：模型R2时间序列、模型R2均值
    '''
    def get_VIF(self,p_threshold)->List[Descriptor]:
        '''
        获取重要的因子(在回归时期中,该因子解释横截面差异占横截面总差异的比例的均值高于n%,且大部分时期p值需小于5%)
        :param p_threshold: 筛选出在这段时期内因子受关注时间点/总时间点的比例达到该值以上的因子(因子受关注:因子收益在该时期显著)
        :return:重要因子列表
        '''
        validCount = 0
        pvaluesCount = np.zeros(self.factors.__len__())
        for i in self.rst:
            ind = self.rst.index(i)
            if(i is not None):
                validCount += 1
                pvaluesCount = pvaluesCount + (i.pvalues<0.05).astype(int)
        AttentionRatio = pvaluesCount / validCount
        return [self.factors[i] for i in range(self.factors.__len__()) if AttentionRatio[i]>p_threshold]

    @property
    def Freturns(self)->pd.DataFrame:
        '''
        获取因子收益序列
        :return:DataFrame，横轴-因子 纵轴-时间
        '''
        returnDict = {}
        for i in range(self.factors.__len__()):
            returnI = []
            for j in self.rst:
                if(j is None):
                    returnI.append(np.nan)
                else:
                    returnI.append(j.params[i])
            returnDict[self.factors[i].descriptorRefName()] = returnI
        factorReturns = pd.DataFrame(index=self.dateIndex,data=returnDict)
        return factorReturns

    @property
    def FreturnCorr(self)->pd.DataFrame:
        '''
        获取因子收益相关系数
        :return:
        '''
        return self.Freturns.corr()

    @property
    def FreturnCov(self)->pd.DataFrame:
        '''
        获取因子收益协方差矩阵
        :return:
        '''
        return self.Freturns.cov()

    @property
    def expFreturn(self):
        '''
        获取预期因子收益
        :return:
        '''
        expFreturn = {}
        for f in self.factors:
            fname = f.descriptorRefName()
            p = self.optimalMAperiod[fname]
            expFreturn[fname] = np.nanmean(self.Freturns[fname].values[-p:])
        return expFreturn


    def save_model(self):
        '''
        存储模型至数据库(做成pickle对象)
        字段:唯一编号；当前时间(精确到秒)；模型二进制BLOB
        '''
        byteObj = pickle.dumps(self)
        timeStr = datetime.now().strftime('%Y%m%d%H%M%S')
        sql = "insert into MultiFactorModels values(?,?)".format(
            Time=timeStr,ModelByte=byteObj)
        self.conn.execute("insert into MultiFactorModels values(?,?)",(timeStr,byteObj))
        self.conn.commit()

    @classmethod
    def load_model(cls):
        '''
        从数据库读取最近一次存储的模型
        :return:最近一次存储的多因子模型对象,为MultiFactor
        '''
        cursor = cls.conn.execute("select model from MultiFactorModels where Time = (select max(Time) from MultiFactorModels)")
        rst = cursor.fetchone()
        if(type(rst) is tuple):
            return pickle.loads(rst[0])
        else:
            print("库内暂未存储多因子模型")
            return None

    def report(self):
        '''
        图形化展示(保存)因子模型报告(写入excel模板)
        todo.
        '''

        pass

    def forcastStkReturn(self):
        '''
        给出下一期股票收益的预测
        :return:dataframe,纵轴:股票，横轴:e_ret,各因子暴露收益
        '''
        nextDate = fapi.TradingTimePoints(asset_code=None, starttime=self.timespan[1] + timedelta(days=1),
                               endtime=self.timespan[1] + timedelta(days=10), freq=self.freq).index[0].to_pydatetime()
        FactorMatDict = {x.descriptorRefName() : get_feature(descriptor=x,starttime=nextDate,endtime=nextDate,freq=self.freq,check=True) for x in self.factors}
        matchDs = xr.Dataset(FactorMatDict)
        CalibratedFeats = {fename:matchDs[fename].values.flatten() for fename in matchDs}
        stocks = matchDs.indexes['StockCode']
        df_factorLoading = pd.DataFrame(index=stocks,data=CalibratedFeats)
        expFreturnLi = [self.expFreturn[f.descriptorRefName()] for f in self.factors]
        df_ForcastReturns = df_factorLoading.mul(expFreturnLi, axis=1)
        df_ForcastReturns['e_ret'] = df_ForcastReturns.sum(axis = 1)
        return df_ForcastReturns

    def bestPortfolio(self,Rf)->pd.Series:
        '''
        根据给定的无风险利率计算切点投资组合.
        todo.
        :param Rf:
        :return:
        '''
        pass