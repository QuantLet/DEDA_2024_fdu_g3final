import pandas as pd
from sklearn.preprocessing import StandardScaler
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# %%
# characteristics of the factors

df_factors = pd.read_csv("stockdata_factors.csv")

# fig = make_subplots(rows=4, cols=1)
# fig.add_trace(go.Scatter(x=df_factors['trade_date'], y=df_factors['adj_close'],
#                          name='Close Price'), row=1, col=1)
# fig.add_trace(go.Scatter(x=df_factors['trade_date'], y=df_factors['MA5'],
#                          name='MA5'), row=1, col=1)
# fig.add_trace(go.Scatter(x=df_factors['trade_date'], y=df_factors['macd'],
#                          name='MACD'), row=2, col=1)
# fig.add_trace(go.Scatter(x=df_factors['trade_date'], y=df_factors['rsi'],
#                          name='RSI'), row=3, col=1)
# fig.add_trace(go.Scatter(x=df_factors['trade_date'], y=df_factors['vwap'],
#                          name='VWAP'), row=4, col=1)
# fig.update_layout(title_text='Technical Indicators')
# fig.show()
# # Save the plot
# fig.write_html("technical_indicators.html")

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(df_factors['trade_date'], df_factors['adj_close'], label='Close Price')
plt.plot(df_factors['trade_date'], df_factors['MA5'], label='MA5')
plt.legend(loc='center left')
# plt.title('Close Price and MA5')

plt.subplot(4, 1, 2)
plt.plot(df_factors['trade_date'], df_factors['macd'], label='MACD')
plt.legend(loc='center left')
# plt.title('MACD')

plt.subplot(4, 1, 3)
plt.plot(df_factors['trade_date'], df_factors['rsi'], label='RSI')
plt.legend(loc='center left')
# plt.title('RSI')

plt.subplot(4, 1, 4)
plt.plot(df_factors['trade_date'], df_factors['vwap'], label='VWAP')
plt.legend(loc='center left')
# plt.title('VWAP')

plt.suptitle('Technical Indicators')
plt.savefig("technical_indicators.png", transparent=True, dpi=400)
plt.show()

# %%
# standardize the factors

df = df_factors.copy()
df.loc[:, 'label'] = df.loc[:, "adj_close"].pct_change().shift(-1)
dflabel = df['label']

plt.figure()
plt.scatter(df['vwap'], df['label'])
plt.savefig('vwap_label.png', dpi=300, transparent=True)
plt.show()

# %%
# Standardize the factors
df = df_factors.copy()
df.set_index('trade_date', inplace=True)

df.loc[:, 'label'] = df.loc[:, "adj_close"].pct_change().shift(-1)
dflabel = df['label']
# set the date as index

# Standardize each column except the 'label' column in the DataFrame
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df),
                               columns=df.columns, index=df.index)

df_standardized = df_standardized.drop(columns=['label'])
df_standardized['label'] = dflabel

df_standardized.dropna(inplace=True)

# get all trade dates in df_standardized, and select them in df

trade_dates = df_standardized.index
df = df.loc[trade_dates]

plt.figure()
plt.scatter(df_standardized['vwap'], df_standardized['label'])
plt.savefig('vwap_label_standardized.png', dpi=300, transparent=True)
plt.show()
