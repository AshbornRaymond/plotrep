import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    X = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    Y1 = [800,700,400,300,100,30,10,100,200,400,600,900]
    Y2 = [190,180,170,180,190,200,180,190,200,170,185,200]
    df = pd.DataFrame({'Month':X,'Jackets':Y1,'Socks':Y2})
    df.plot(kind='line',x='Month',y=['Jackets','Socks'],title='Revenue - Jackets and Socks',legend=True)
    plt.legend(loc='center')
    plt.xlabel('Month')
    plt.ylabel('Prices')
    plt.xticks(np.arange(0,12,1),labels=X)
    plt.yticks(np.arange(100, 1000, 100),label=Y1)
    plt.show()
main()
