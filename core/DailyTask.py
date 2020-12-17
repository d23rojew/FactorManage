'''
定时任务:
每日日终:
    1、增量(非自检模式)刷新行情数据
    2、获取模型栈中本日激活模型清单，更新清单中模型的因子数据
    3、更新清单中的因子模型
    4、调用modelAssembler,综合模型周期及数据库的仓位记录，得出最新持仓目标、模型应结仓位并记录入库
每日日初：
    5、调用DecisionMaker，从数据库中获取最新持仓目标，与账户实际数据对比后下单(同时反馈开仓情况、平仓情况入库)
       ----------------
       多周期模型的嵌套调仓:
       ----------------
            为充分利用各个时间周期的异常定价机会，可构造多个周期的多因子预测模型，短周期模型可
       修正长周期模型的仓位以提高短期边际收益贡献。长周期必须为短周期的整数倍，并且这个修
       正调仓可以嵌套。
            下面假定我们的模型组为:日模型、周模型、双周模型、四周模型
            .每四周、每双周、每周、每日定期运行该周期的预测模型，并进行相应调仓
            .每次调仓会记录该笔调仓的模型来源、调整日期及调整向量w_adj.
            .每次调仓的优化目标是它预测周期本身的风险调整后、父周期机会成本调整后收益，具体为:
            max z = 【周期增量收益率】w_adj * r_t
                   -【周期增量风险】lambda*(risk_after-risk_before)
                   -【复原父周期仓位交易成本】w_adj * fee_Rate
            s.t:
                0 < w_fathernow + w_adj < I;
                w_adj*I = 0;
            注:
                w_fathernow:假设不进行调仓,父周期初始仓位w_origin随行情变化至当时的仓位
                w_adj:目标仓位相对于父周期仓位w_fathernow的调整量
                risk_before:w_fathernow权重下,子周期中组合收益标准差
                risk_after:调整后(w_fathernow + w_adj)权重下,子周期中组合收益标准差
            在底层周期模型需调仓时，簿记前底层周期模型区间实现收益;将(|w_adj_new|+|w_adj_old|-|w_adj_new-w_adj_old|)部分的
            收益归属于前底层周期
            在父周期模型需要调仓时，子周期模型的开放仓位从低到高逐级关闭(仓位回归至w_fathernow);由此产
            生手续费共为(假定该周期层级为3):
                feerate*(|w_adj_old_l1|+|w_adj_old_l2|+|w_adj_old_l3|)
            然后再展开放新一期调整
                feerate*(|w_adj_new_l3|+|w_adj_new_l2|+|w_adj_new_l1|)
            但交易之间会相互抵消，因此实际的手续费为
                feerate*|w_adj_new_l3+w_adj_new_l2+w_adj_new_l1-w_adj_old_l1-w_adj_old_l2-w_adj_old_l3|
            将该手续费节省全部归属给上个父周期模型。
'''
#可以开始写每日日程代码了
#备好后进行单元测试

import core.fundamentals.DataCollectionImpls.UpdateData as upd
from core.fundamentals.getfundamentals import fundamentalApi as fApi
from datetime import datetime
from core.model.ModelAssembler import DecisionMaker
from core.model.MultiFactor import MultiFactor
#刷新行情数据
# ----------------------------
# upd.renewTradingDays(fApi.conn)
# trade_date = fApi.trade_cal('SSE', datetime.now().strftime('%Y%m%d'), datetime.now().strftime('%Y%m%d'), '1')
# checkAll = trade_date.__len__()<0 #今日是否为休息日
# upd.renewall(fApi.conn,checkAll) #休息日更新时自检，交易日更新时仅更新当天数据，不自检
#更新模型并入库
ModelList = [x[1] for x in DecisionMaker.getActiveModelStackFromDB()]
for model in ModelList:
    model:MultiFactor
    model.uptodate()
#使用模型下预置指令
DecisionMaker.makeOptimizeOrder()
#次日开盘，使用模型下单并记录结果
DecisionMaker.fire()