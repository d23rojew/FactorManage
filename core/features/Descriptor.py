'''
author: dongruoyu
time: 2020-02-14
为feature计算类定义模板.
'''
import abc
from collections import OrderedDict
class Descriptor(abc.ABC):
    '''
    规范:编写特征的具体算法需实现该类,实现类的【名字】必须为特征名
    Descriptor的实现类由getfeature调用,在get_featureFromDB函数发现数据库中股票特征数据有缺失时,根据缺失特征的【名字】查找
    对应的Descriptor实现类,调用其calcDescriptor方法获取特征值
    '''

    calcMinDescriptor,calcDayDescriptor,calcWeekDescriptor,calcMonthDescriptor,FixDescriptor = None,None,None,None,None
    '''
    计算分钟/日/周/月/固定特征的方法，子类需覆盖其中至少一个方法。
    '''

    category = {}
    '''
    标注适用类别的集合。可包含"stock"、"commodity"、"index".合约只能获取与其适用的特征，若
    非适用特征则返回np.nan。
    被get_feature用于判定所取标的与特征是否匹配.
    '''

    def __init__(self, params={}, freq=None, prepareProc=None,ctrlDescriptorList=[]):
        if(freq is not None):
            self.freq = freq
        if("stock" not in self.category and (prepareProc is not None or ctrlDescriptorList.__len__()>0)):
            print("特征" + type(self).__name__ + "非股票特征，无法设置预处理或控制特征")
            self.prepareProc = None
            self.ctrlDescriptorList = []
        if(type(prepareProc) is str):
            self.prepareProc = [prepareProc]
        elif(type(prepareProc) is list):
            self.prepareProc = prepareProc
        if(type(ctrlDescriptorList) is not list):
            self.ctrlDescriptorList = []
        else:
            for ctrl in ctrlDescriptorList:
                if(type(ctrl) not in Descriptor):
                    raise Exception("控制因子列表 ctrlDescriptorList 的元素必须为Descriptor对象!")
            self.ctrlDescriptorList = ctrlDescriptorList
        self.params = params
        self.SQLName = self.descriptorSQLName()

        self.calcDescriptor = OrderedDict()
        '''
        这是一个五元有序字典,装载了不同时间频率('min'/'d'/'w'/'m'/'fix')下特征的计算方法,方法签名类似calc(self, stock_code:str, timepoint:str)
        该方法对给定的股票代码、时间点、参数设置返回一个特征值(字符或数值),或者None(将转化为sqlite中的null)
        计算k时点的特征会用于预测k时点的收益，因此特征必只能用k-1时点及以前的信息
        '''
        self.calcDescriptor['min'] = self.calcMinDescriptor
        self.calcDescriptor['d'] = self.calcDayDescriptor
        self.calcDescriptor['w'] = self.calcWeekDescriptor
        self.calcDescriptor['m'] = self.calcMonthDescriptor
        self.calcDescriptor['fix'] = self.FixDescriptor

    prepareProc = []
    '''
    数据预处理列表。列表中元素可为:
    'winsor':横截面样本中超出1%~99%分位数的特征，以1%和99%分位数替代
    'ln'：对样本取对数
    'std'：将样本横截面标准化均值为0，方差1
    prepareProc将被get_feature函数解读，并从第一项开始进行预处理.
    '''

    params = {}
    '''
    特征参数字典
    '''

    freq = ''
    '''
    特征频率('d'/'w'/'m')
    '''

    ctrlDescriptorList = []
    '''
    特征的控制特征
    说明：部分特征是通过粗特征对其它特征横截面回归取残差后获得的。对于这部分特征，其单时间点、单股票可计算部分
    由calcDescriptor负责，需要整体回归取残差部分由get_feature负责。
    在计算带有ctrlDescriptorList的特征值时，先计算不加控制的特征并存储，get_feature再次取出存储的特征并整体回归/
    取残差入库，标签需带上控制的因子。
    '''

    SQLName = __name__
    '''
    标记了参数的特征名，用于数据库存储
    '''

    def descriptorSQLName(self):
        # 用Descriptor名、参数、控制列表得出顺序固定的字符串,用于数据库检索
        wait4sortParams = []
        if (self.params != None and self.params != {}):
            for k, v in self.params.items():
                if (type(k) != str):
                    raise Exception('参数字典{paramdict}异常:feature {feature}参数名必须为字符串!'.format(paramdict=str(self.params),
                                                                                           feature=self.__class__.__name__))
                if (type(v) not in (int, float, str)):
                    raise Exception('参数字典{paramdict}异常:feature {feature}参数值必须为字符串或数值!'.format(paramdict=str(self.params),
                                                                                              feature=self.__class__.__name__))
                if (type(v) is float and int(v) == v):
                    v = int(v)
                wait4sortParams.append((k, v))
            wait4sortParams.sort()
        sortedDict = {tuple[0]: tuple[1] for tuple in wait4sortParams}
        wait4sortCtrlDescriptor = [ctrlDesc.descriptorRefName() for ctrlDesc in self.ctrlDescriptorList]
        sortedCtrlDescriptor = wait4sortCtrlDescriptor.sort()
        fename = self.__class__.__name__ + '_Params:' + str(sortedDict) + '_Contrls:' + str(sortedCtrlDescriptor)
        return fename.replace('\'', '')

    def descriptorRefName(self):
        # 用Descriptor名、参数、控制列表、预处理方法得出顺序固定的字符串,用于在作为控制因子时，给回归特征命名
        fename = self.SQLName + '_prepareProc:' + str(self.prepareProc.sort())
        return fename.replace('\'', '')