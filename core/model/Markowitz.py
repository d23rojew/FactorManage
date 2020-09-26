'''
输入:现有持仓权重+股价+调仓费率+固定费用
     股票因子暴露、因子预期收益、因子协方差、最新股价
     在此基础上，给出增量收益最大、增量风险最小的调整方案
输出:还是以权重的形式输出
可归因性:
       权重优化器的每次调仓，返回调仓权重、调仓成本、预期收益
'''
import pandas as pd
import numpy as np
import cvxpy as cvx
def Markowitz(stk_feature:pd.DataFrame,stk_price:pd.Series,stk_hold:pd.Series,f_return:pd.DataFrame,f_cov:pd.DataFrame,feeRate:float,totalNV):
    '''
    根据当前账户总量、持股比例、股票信息【现价、因子暴露】、因子信息【预期收益，协方差矩阵】、费率
    计算最佳调整量
    :param stk_feature:股票因子暴露【cols:因子暴露factor1,因子暴露factor2,...;rows:stock_code】
    :param stk_price:股票价格【cols:股票价格price,rows:股票代码】
    :param stk_hold:股票持仓比例【cols:持仓权重,rows:股票代码】
    :param f_return:因子收益率
    :param f_return:因子协方差
    :param fee:交易费率
    :param totalMV:账户总净值
    :return:调仓权重、调仓成本、边际收益贡献、边际风险贡献
    '''
    #我他妈先验证特征数齐不齐,特征数不对优化个球
    k = f_return.__len__()
    if(k!=stk_feature.shape[1]):
        raise Exception("stk_feature股票特征数为{}，与f_return特征收益数{}不匹配!".format(k,stk_feature.shape[1]))
    if(k!=f_cov.shape[0] or k!=f_cov.shape[1]):
        raise Exception("stk_feature股票特征数为{}，与f_cov特征协方差维数[{},{}]不匹配!".format(k,f_cov.shape[0],f_cov.shape[1]))
    # 股票信息对齐
    stk_price.name = 'Price'
    stk_hold.name = 'Hold'
    stockInfo = stk_feature.merge(stk_price,how="outer",left_index=True,right_index=True)\
    .merge(stk_hold,how="outer",left_index=True,right_index=True)
    # n只股票，k个特征
    n = stockInfo.__len__()
    n=200
    F = stockInfo.values[0:n,0:k]       # n*k feature Matrix
    F[np.isnan(F)] = 0
    R_F = f_return.values         # k*1 feature return
    P = stockInfo.values[0:n,k]          # n*1 stock Price
    P[np.isnan(P)] = 20
    H = stockInfo.values[0:n,k+1]          # n*1 stock Hold(old)
    H[np.isnan(H)] = 0
    Sigma_F = f_cov.values        # k*k feature return covariance
    w = cvx.Variable(n,integer=True) # 调整手数(100股/手)
    adjAmount = 100 * cvx.multiply(w, P) #调整金额
    expRet = cvx.matmul(adjAmount.T,F.dot(R_F)) #预期收益
    risk = cvx.quad_form(cvx.matmul(F.T,adjAmount),Sigma_F) #系统风险
    gamma = 0.1     #风险厌恶系数
    tradefee = cvx.sum(cvx.abs(adjAmount)*feeRate)         #交易费用(未考虑单笔最低费用)
    # target = cvx.Maximize(expRet - gamma*risk - tradefee)
    target = cvx.Maximize(expRet- tradefee)
    #注意条件中的nan（尤其持仓H或价格P，容易造成问题无解）
    constrant_1 = w*100 + H >= 0      # 不能卖空任何股票
    constrant_2 = cvx.matmul(P,w*100 + H) <= totalNV;  #总资产约束
    prob = cvx.Problem(target,[constrant_1,constrant_2])
    prob.solve(solver='ECOS_BB',max_iters=1000)
    return 1

