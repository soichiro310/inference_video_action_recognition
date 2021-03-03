# inference_video_action_recognition
## demo
![](./doc/demo_mac_safari.gif)

## 使用言語，フレームワーク
詳細は[`requirements.txt`](https://github.com/soichiro310/inference_video_action_recognition/blob/develop/requirements.txt)を参照．
* Python 3.7.6
* PyTorch 1.7.1
* Flask 1.1.2
* OpenCV 4.5.1
* Pillow 8.1.1

## setup
* venvを使用した仮想環境を構築し，そこに必要なパッケージをダウンロードする．
* 今回は，cpu推論のみを行う仮想環境の構築方法について説明する．
* NVIDIA製GPUを搭載したマシン向けの仮想環境構築は後日動作確認し，記載する予定．
### cpu only
1. このリポジトリをクローン
    ```
    git clone https://github.com/soichiro310/inference_video_action_recognition
    ```
2. venvを使用して仮想環境を構築 (今回はcpu_inferenceにしたが，名前は任意で指定可能．)
    ```
    cd inference_video_action_recognition
    python3 -m venv cpu_inference
    ```
3. 仮想環境をアクティベート
    ```
    source cpu_inference/bin/activate
    ```
4. パッケージのインストール
    ```
    pip install -r requirements.txt
    ```
## 実行方法
* `python3 run.py`を実行すれば，[localhost:8888](http://localhost:8888/)にFlaskサーバーが立ち上がる．
* また，コマンドライン引数にて以下の設定を行うことができる
    * `--host`：IPアドレスを指定．
    * `--port`：ポートを指定．(デフォルト値は`8888`)
    * `--sample_video_dir`：プルダウンメニューで指定する動画ファイル(.avi)を格納したディレクトリのパスを指定．
    * `--device`：推論を行うデバイスの指定．cpuによる推論の場合は`cpu`，gpuによる推論の場合は`cuda:0`で指定すれば良い．(デフォルト値は`cpu`)
### 実行例
* ipアドレスを`133.15.12.123`，サンプル動画を`./sample_video`に格納，NVIDIA製GPUによる推論を行う場合
    ```
    python3 run.py --host 133.15.12.123 --sample_video_dir ./sample_video --device cuda:0
    ```
## 出典
* [Kineticsデータセットによる事前学習済みI3Dモデル(rimchang/kinetics-i3d-Pytorch)](https://github.com/rimchang/kinetics-i3d-Pytorch)
