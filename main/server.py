from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
import keras
import librosa
from keras.models import load_model
from pydub import AudioSegment

# web アプリケーションフレームワーク Flask
app = Flask(__name__)

# グローバル変数を宣言
global model, autotext

# 使うモデルを指定
model = load_model("../../mini7.hdf5")

# 学習モデルを読み込む関数
def load_model(x):
    model = load_model(x)

# 波形データ -> メルスペクトログラム 変換関数
def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    return melsp


@app.route('/', methods=["GET", "POST"])    # http://localhost:5000 と index() を紐付け
def index():                                # http://localhost:5000 にアクセスしたら以下を実行
    return render_template('index.html')    # index.html を出力


@app.route('/result', methods=['POST'])
def result():                               # # /result にアクセスしたら以下を実行
    # 解析するファイル指定
    recieive2html = "/Users/xxx/Downloads/test.wav"
    # データロード
    sound = AudioSegment.from_wav(recieive2html)
    sound1 = sound[:3000]
    sound1.export(recieive2html, format="wav")
    x, fs = librosa.load(recieive2html, sr=44100)
    # メルスペクトラム取得
    melsp = calculate_melsp(x)
    # 四次元テンソル化
    melsp = melsp.reshape(128,1034,1)
    melsp = melsp.reshape(1,128,1034,1)
    # 各ラベルに対する確信度
    predictions = model.predict(melsp)
    # 確信度の最も高いラベルを出力
    result = predictions.argmax()
    # ラベル番号（配列）から .item()で要素の値（整数）に変換
    FontNo = result.item()

    # 変数 FontNo を result.html に渡して出力
    return render_template("result.html", FontNo=FontNo)


if __name__ == '__main__':
    app.debug = True
    app.run(host='localhost', port=5000)    # http://localhost:5000 を立てる（'/'になるページ）
