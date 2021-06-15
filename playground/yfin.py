import yfinance as yf

msft = yf.Ticker("^FVX")

# get stock info
print(msft.info)
print(dir(msft))

print(msft.history(period='5y', interval='1mo'))
quit()

# get historical market data
# hist = msft.history(period="max")

# show actions (dividends, splits)
# msft.actions



# show splits
# msft.splits

# show financials
b = msft.financials
a = msft.get_balance_sheet(proxy="PROXY_SERVER")


# show dividends
c = msft.dividends

print(a)
print('----')
print(b)
print('----')
print(c)