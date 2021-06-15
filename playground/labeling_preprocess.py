import pandas as pd
import pandas_ta as ta
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor, RadioButtons, Button
import matplotlib.dates as mdates
import numpy as np
import  datetime



x_sell = []
y_sell = []
x_buy = []
y_buy = []
def on_click(event):
    
    # GET THE CLICKED DATE
    selected_date = mdates.num2date(event.xdata)
    selected_date = selected_date.replace(tzinfo=None)
    if str(selected_date.year) == '1970':
        return
    
    # FIND CLOSEST IN DF
    _df = df.iloc[df.index.get_loc(selected_date,method='nearest')]
    selected_close = _df.close
    # print('you pressed', selected_date.strftime('%Y-%m-%d'))

    if event.key=='shift' and (event.button is MouseButton.LEFT or MouseButton.RIGHT):
        try:


            if event.button is MouseButton.LEFT:
                x_buy.append(_df.name)
                y_buy.append(selected_close)
                print(f'Buy added at {_df.name}')
                
            elif event.button is MouseButton.RIGHT:
                x_sell.append(_df.name)
                y_sell.append(selected_close)
                print(f'Sell added at {_df.name}')

        except:
            pass
        

    elif event.key == 'control' and (event.button is MouseButton.LEFT or MouseButton.RIGHT):
        
        try:
            idx = y_sell.index(selected_close)
            print(f'Sell removed at {x_sell[idx]}')
            x_sell.pop(idx)
            y_sell.pop(idx)
            
        except:
            pass

        try:
            idx = y_buy.index(selected_close)
            print(f'Buy removed at {x_buy[idx]}')
            x_buy.pop(idx)
            y_buy.pop(idx)
            
        except:
            pass



    ax1.clear()
    ax1.plot(df.index, df[['close']], linestyle='-', marker='')
    ax1.scatter(x_sell,y_sell, color='r')
    ax1.scatter(x_buy,y_buy, color='g')
    plt.sca(ax1)
    plt.title('VALLEY:[SHIFT+LEFT] - PEAK:[SHIFT+RIGHT] - HOLD CTRL TO REMOVE')
    plt.draw() #redraw


tick = 'TSLA'
year = '2011'
df = pd.read_csv(tick+'.csv')
df.set_index('Date', inplace=True)
df.index = pd.to_datetime(df.index)

df.ta.macd(fast=12, slow=26, append=True)
df.ta.rsi(append=True)



df = df[year]




''' PLOTS '''
fig, (ax1, ax2, ax3) = plt.subplots(3,1)

ax1 = plt.subplot(3,1,1)
ax1.plot(df.index, df[['close']], linestyle='-', marker='')
plt.title('VALLEY:[SHIFT+LEFT] - PEAK:[SHIFT+RIGHT] - HOLD CTRL TO REMOVE')

ax2 = plt.subplot(3,1,2, sharex=ax1)
plt.plot(df.index, df[['MACD_12_26_9', 'MACDs_12_26_9']])

ax3 = plt.subplot(3,1,3, sharex=ax1)
plt.plot(df.index, df[['RSI_14']])

sell_scatter = ax1.scatter([np.nan],[np.nan], color='r', label='sell')

cid = fig.canvas.mpl_connect('button_press_event', on_click)





class Menu:
    ind = 0

    def done(self, event):
        print('Done')

        df['target'] = [[1,0,0]]*len(df)
        for x in x_buy:
            df.target.loc[x] = [0,1,0]
        for x in x_sell:
            df.target.loc[x] = [0,0,1]
        print(df)
        df.to_csv('processed/'+f'{tick}_{year}_processed.csv' ,index=True)

    # def prev(self, event):
    #     print('awd')

callback = Menu()
# axprev = plt.axes([0.7, 0.0, 0.1, 0.075])
axdone = plt.axes([0.81, 0.0, 0.1, 0.075])
button_done = Button(axdone, 'Save')
button_done.on_clicked(callback.done)
# bprev = Button(axprev, 'Previous')
# bprev.on_clicked(callback.prev)





multi = MultiCursor(fig.canvas, (ax1, ax2, ax3), color='r', lw=1)
# ax1.set_xlim([datetime.date(2020, 1, 1), datetime.date(2020, 10, 1)])


plt.subplots_adjust(bottom=0.18)
plt.show()





