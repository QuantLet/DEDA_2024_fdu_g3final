import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

df_factors = pd.read_csv("stockdata_factors.csv")
df_factors.dropna(inplace=True)
df_factors.loc[:, 'label'] = df_factors.loc[:,
                                            "adj_close"].pct_change().shift(-1)
X = np.array(df_factors.drop(columns=['label']))
X = np.array(df_factors.drop(columns=['trade_date']))
X = preprocessing.scale(X)
y = np.array(df_factors['label'])

train_test_split_idx = int(len(X) * 0.8)

train_df = df_factors.loc[:train_test_split_idx].copy()
test_df = df_factors.loc[train_test_split_idx:].copy()

plt.figure(figsize=(12, 6))

plt.plot(train_df['trade_date'], train_df['adj_close'], label='Train')
plt.plot(test_df['trade_date'], test_df['adj_close'], label='Test')

plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.title('Train/Test Split')
plt.legend()

N = 100
xticks = train_df['trade_date'][::N].tolist() \
    + test_df['trade_date'][::N].tolist()
plt.xticks(ticks=xticks, rotation=45)
plt.tight_layout()

plt.savefig('train_test_split.png', dpi=300, transparent=True)
plt.show()
