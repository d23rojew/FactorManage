from datetime import datetime,timedelta

def getFirstDate(date:datetime,freq:str):
    if(freq=='w'):
        weekday = date.weekday()
        return date-timedelta(days=weekday)
    elif(freq=='M'):
        return datetime.strptime(date.strftime('%Y%m'),'%Y%m')
    else:
        raise Exception("当前仅支持周度频率!")

