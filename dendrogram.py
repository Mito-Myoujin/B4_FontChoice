#B3前期のアンケートを階層的クラスタリング（ウォード法*ユークリッド距離）
#結果をデンドログラムで表示する

# 参考記事：購買データを使って顧客のクラスター分析を行ってみた
#       https://qiita.com/oka_1207/items/d49fffb3d31018ec1852
# 【一瞬の興味】日本人はどこの国の国民性と似ているか
#       https://qiita.com/pear_0/items/dd8a56465829055a9804
# 【python】scipyで階層型クラスタリングするときの知見まとめ
#       https://www.haya-programming.com/entry/2019/02/11/035943#dendrogram

import pandas as pd
import numpy as np

df = pd.read_csv('FontData2.csv', index_col=0)
#print(df)
#print(df.index)

df.data = df

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# linkage で階層クラスタリング
# ウォード法*ユークリッド距離
linkage_result = linkage(df.data, method='ward', metric='euclidean')

# クラスター分けするしきい値
threshold2 = 0.7 * np.max(linkage_result[:, 2])

# 表示
plt.figure(num=None, figsize=(16, 9), dpi=200, facecolor='w', edgecolor='k')
dendrogram(linkage_result, labels=df.index, orientation="left", color_threshold=threshold2)
plt.show()
