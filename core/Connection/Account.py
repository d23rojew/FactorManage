from easytrader.yh_clienttrader import YHClientTrader
import easytrader as et
from time import time

class Account:
    '''
    该类包含了所有与交易端相关的操作。
    '''
    accountNo = '70043707'
    passWord = '987748'
    user:YHClientTrader = None

    def __init__(self):
        pass

    @classmethod
    def __makeSureConnect(cls):
        '''
        一些确保账户连接可用的操作
        :return:
        '''
        count=0
        while count<3:
            try:
                #验证连接
                cls.user.balance
                break
            except:
                count += 1
                print('账户未连接，现在开始执行第{}次连接操作!'.format(str(count)))
                try:
                    cls.user = et.use('yh_client')
                    cls.user.prepare(user='202100007679', password='987748', exe_path='D:/双子星金融终端独立交易-中国银河证券/xiadan.exe')
                    cls.user.connect(exe_path='D:/双子星金融终端独立交易-中国银河证券/xiadan.exe')
                except:
                    time.sleep(1)
                    pass

    @classmethod
    def getNav(cls)->float:
        '''
        获取总资产
        :return: 总资产
        '''
        cls.__makeSureConnect()
        return cls.user.balance[0].get('总资产')

    @classmethod
    def placeOrder(cls,direction:str,stockcode:str,amount:int)->bool:
        '''
        下市价单
        :param direction:方向:买->buy 卖->sell
        :return:成交反馈:成功->true 失败->false
        '''
        cls.__makeSureConnect()
        try:
            if(direction=='buy'):
                cls.user.market_buy(stockcode,amount)
            elif(direction=='sell'):
                cls.user.market_sell(stockcode, amount)
            else:
                raise Exception("下单方向{dir}错误！仅支持buy和sell".format(dir=direction))
            return True
        except:
            return False

    @classmethod
    def getPosition(cls):
        '''
        获取持仓
        :return:
        '''
        cls.__makeSureConnect()
        return cls.user.position

    @classmethod
    def getStockPosition(cls,stkcode:str)->int:
        '''
        获取单只股票持仓
        :return:股票持仓量
        '''
        cls.__makeSureConnect()
        return cls.user.position[stkcode]

