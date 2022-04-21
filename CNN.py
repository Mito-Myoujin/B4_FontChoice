import os
import numpy as np
import keras
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Conv2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
#from keras.optimizers import Adam
from keras.models import load_model



# データセットを指定
train_files = ["npz/esc_melsp_train_raw.npz",
               "npz/esc_melsp_train_ss.npz",
               "npz/esc_melsp_train_st.npz",
               "npz/esc_melsp_train_wn.npz",
               "npz/esc_melsp_train_com.npz"]

test_file = "npz/esc_melsp_test.npz"

train_num = 120     # 訓練データ数
test_num = 41       # テストデータ数(DataChage.pyあたりで確認できる)
freq = 128          # 周波数
time = 1034         # ファイルの長さ(time)指定



# 各データセット用placeholderの定義
x_train = np.zeros(freq*time*train_num*len(train_files)).reshape(train_num*len(train_files), freq, time)
y_train = np.zeros(train_num*len(train_files))

# 学習データのロード
for i in range(len(train_files)):
    data = np.load(train_files[i])
    x_train[i*train_num:(i+1)*train_num] = data["x"]
    y_train[i*train_num:(i+1)*train_num] = data["y"]
# テストデータのロード
test_data = np.load(test_file)
x_test = test_data["x"]
y_test = test_data["y"]

# ラベルをバイナリクラス行列に変換(one-hot表現)
classes = 6       # クラス数
y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)

# 学習データを(batch_size, freq, time, 1)に変換
x_train = x_train.reshape(train_num*5, freq, time, 1)
x_test = x_test.reshape(test_num, freq, time, 1)

print("x train:{0}\ny train:{1}\nx test:{2}\ny test:{3}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

# 【長さの違う複数のフィルタで畳込する関数】
def cba(inputs, filters, kernel_size, strides):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
        # Conv2D：2次元の畳み込みレイヤー
        #       filters：フィルタ（カーネル/特徴検出器）のサイズ
    x = BatchNormalization()(x)
        # BatchNormalization：各バッチ毎に前の層の出力（このレイヤーへの入力）を正規化
    x = Activation("relu")(x)
        # Activation：出力に活性化関数を適用
    return x


inputs = Input(shape=(x_train.shape[1:]))
x_1 = cba(inputs, filters=32, kernel_size=(1,8), strides=(1,2))
x = cba(x_1, filters=128, kernel_size=(1,16), strides=(1,2))
x = cba(x, filters=128, kernel_size=(16,1), strides=(2,1))

x = GlobalAveragePooling2D()(x)     # GlobalAveragePoolingして分類
x = Dense(classes)(x)
x = Activation("softmax")(x)        # 出力層の活性化関数：softmaxm関数
model = Model(inputs, x)



# 【Adam】
#opt = keras.optimizers.Adam(lr=0.1, decay=1e-6, amsgrad=True)
    #   lr = 学習率
    #   decay = 0以上の浮動小数点数．各更新の学習率減衰．
    #   amsgrad = 論文"On the Convergence of Adam and Beyond"にあるAdamの変種であるAMSGradを適用するかどうか．
    #   beta_1 = 浮動小数点数, 0 < beta < 1. 一般的に1に近い値です．
    #   beta_2 = 浮動小数点数, 0 < beta < 1. 一般的に1に近い値です．

# 【Adam】提案論文でのパラメータ
opt = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# 【SGD：確率的勾配降下法オプティマイザ】
#opt = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    #   lr = 0以上の浮動小数点数．学習率．
    #   momentum = 0以上の浮動小数点数．モーメンタム．
    #   nesterov = 真理値. Nesterov momentumを適用するかどうか．

# 【RMSprop】 デフォルトパラメータのまま利用することを推奨(学習率は自由に調整可能）
#opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    # RMSPropはリカレントニューラルネットワークに対して良い選択となるでしょう．
    #   rho = 0以上の浮動小数点数．
    #   epsilon = 0以上の浮動小数点数．微小量．NoneならばデフォルトでK.epsilon()

# 【Adagrad】 デフォルトパラメータのまま利用することを推奨
#opt = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

# 【Adadelta】 デフォルトパラメータのまま利用することを推奨
#opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

# 【Adamax】 無限ノルムに基づくAdamの拡張　デフォルトパラメータは提案論文に従う
#opt = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

# 【Nadam：Nesterov Adamオプティマイザ】デフォルトパラメータのまま利用することを推奨
#opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)



# 学習処理の設定
model.compile(loss = 'categorical_crossentropy',  # 損失関数指定：交差エントロピー誤差(多クラス分類)
              optimizer = opt,                    # パラメタ最適化のための探索アルゴリズム指定
              metrics = ['accuracy'])             # 評価を行うためのリストの方式指定？

# モデルの要約を出力
#model.summary()
# モデルの可視化
#keras.utils.plot_model(model, to_file='model.svg', show_shapes=True)
# モデルを .pig で出力
from keras.utils import plot_model
plot_model(model, to_file='model.png')

# コールバックするチェックポイント保存用ディレクトリ作成
model_dir = "./models"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
#chkpt = os.path.join(model_dir, 'esc50_.{epoch:02d}_{val_loss:.4f}_{val_acc:.4f}.hdf5')
chkpt = os.path.join(model_dir, 'label.hdf5')
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')


# train model
batch_size = 64     # 同時計算するサンプル数
epochs = 500           # 学習サイクル数

# 訓練開始
fit = model.fit(x_train,                      # 訓練データのNumpy配列
                y_train,                      # ラベルデータのNumpy配列
                batch_size = batch_size,      # 設定したサンプル数ごとに勾配を更新
                epochs = epochs,
                validation_data=(x_test, y_test), # 評価用データ指定
                callbacks = [es_cb, cp_cb])   # コールバック



# モデル保存
model.save( "model.hdf5" )
# モデル読み込み
model = load_model("model.hdf5")
# 学習済モデルの評価
evaluation = model.evaluate(x_test, y_test)    # evaluate：テストデータを指定することで損失関数と評価関数の結果を返す
# 結果表示
test_loss, test_acc = evaluation
print('loss：%f' %test_loss)
print('acc：%f' %test_acc)

# グラフ化
import matplotlib.pyplot as plt
metrics = ['loss', 'accuracy']  # 使用する評価関数を指定
plt.figure(figsize=(10, 5))  # グラフを表示するスペースを用意
for i in range(len(metrics)):
    metric = metrics[i]
    plt.subplot(1, 2, i+1)  # figureを1×2のスペースに分け、i+1番目のスペースを使う
    plt.title(metric)  # グラフのタイトルを表示
    plt_train = fit.history[metric]  # historyから訓練データの評価を取り出す
    plt_test = fit.history['val_' + metric]  # historyからテストデータの評価を取り出す
    plt.plot(plt_train, label='training')  # 訓練データの評価をグラフにプロット
    plt.plot(plt_test, label='test')  # テストデータの評価をグラフにプロット
    plt.legend()  # ラベルの表示

plt.show()  # グラフの表示
