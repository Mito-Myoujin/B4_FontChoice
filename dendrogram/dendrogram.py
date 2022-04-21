import pandas as pd
import numpy as np

df = pd.read_csv('FontData.csv', index_col=0)

df.data = df

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# linkage で階層クラスタリング
# ウォード法*ユークリッド距離
linkage_result = linkage(df.data, method='ward', metric='euclidean')

# 閾値
threshold2 = 0.7 * np.max(linkage_result[:, 2])

# 表示
plt.figure(num=None, figsize=(16, 9), dpi=200, facecolor='w', edgecolor='k')
dendrogram(linkage_result, labels=df.index, orientation="left", color_threshold=threshold2)
plt.show()
