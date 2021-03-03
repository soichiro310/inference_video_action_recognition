# inference_video_action_recognition
## demo
![](./doc/demo_mac_safari.gif)

## 使用言語，フレームワーク
* Python 3.7.6
* PyTorch 1.1.0
* Flask 1.1.2

## setup
venvを使用した仮想環境を構築し，そこに必要なパッケージをダウンロードする．
今回は，cpu推論のみを行う仮想環境の構築方法について説明する．
(NVIDIA製GPUを搭載したマシン向けの仮想環境構築は後日動作確認し，記載する予定)
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


## 出典
* [Kineticsデータセットによる事前学習済みI3Dモデル(rimchang/kinetics-i3d-Pytorch)](https://github.com/rimchang/kinetics-i3d-Pytorch)
