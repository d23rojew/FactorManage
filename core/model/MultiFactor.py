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
3、确定日常模块
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


日内：
暂时不去考虑
'''
from datetime import datetime
from core.fundamentals.getfundamentals import fundamentalApi as fapi
from core.features.DescriptorImpls import *
from core.features.getfeature import get_feature
import xarray as xr
from linearmodels import FamaMacBeth
class MultiFactor:
    def __init__(self,factors:list,starttime:datetime,endtime:datetime,forcast_period:int,controls:list,freq:str='d'):
        '''
        :param factors: 参与建模的因子列表，类型:list<Descriptor>
        :param starttime: 建模开始时间
        :param endtime: 建模结束时间
        :param forcast_period: 模型预测期(形如'1d'/'2w')
        :param controls: 控制因子列表，类型:list<Descriptor>，个股在该时间点控制因子值为0则该条数据不参与横截面回归
        '''
        #取收益矩阵
        ReturnMat = fapi.getReturn(asset_code=None,starttime=starttime,endtime=endtime,forcast_period=forcast_period,freq=freq,asset_type='stock')
        #取因子矩阵
        FactorMatDict = {'feat_'+x.descriptorRefName() : get_feature(descriptor=x,starttime=starttime,endtime=endtime,freq=freq,check=False) for x in factors}
        ControlMatDict = {'ctrl_'+x.descriptorRefName() : get_feature(descriptor=x,starttime=starttime,endtime=endtime,freq=freq,check=False) for x in controls}
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
        model = FamaMacBeth(MaskedReturn,np.stack(CalibratedFeats))
        self.report = model.fit(cov_type='kernel') #由于存在重叠样本，使用newey west调整后的t值

    report = {}
    '''
    报告项:
    单因子相关: 因子收益序列(已控制其它因子的)
    因子全貌: 因子暴露相关性矩阵、因子收益协方差矩阵、因子收益序列均值(t值)、因子IC/IR均值
    模型质量：模型R2时间序列、模型R2均值
    '''

    def get_VIF(self):
        '''
        获取最重要的因子
        :return:
        '''
        pass

    def get_Freturns(self):
        '''
        获取因子收益序列
        :return:
        '''

    def get_FreturnCov(self):
        '''
        获取因子收益协方差矩阵
        :return:
        '''
        pass

    def save_model(self):
        '''
        存储模型至数据库(做成pickle对象)
        :return:
        '''
        pass

    @classmethod

    def load_model(cls):
        '''
        从数据库读取模型
        :return:
        '''
        pass

    def plot_report(self):
        '''
        图形化展示因子模型报告
        '''
        pass