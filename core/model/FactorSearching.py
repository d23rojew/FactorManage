'''
使用遗传算法&多因子模型评价框架，演进寻找量价因子
deap包负责公式树生成、演进，输出公式字符串
FomulaDailyFactor类接收公式字符串为参数，负责实质计算
MultiFactor负责生成因子的评价
'''
import deap.gp as gp
from core.features.DescriptorImpls import FomulaDailyFactor as fclass
pset = gp.PrimitiveSet("main",2)
pset.addPrimitive("rank",1)