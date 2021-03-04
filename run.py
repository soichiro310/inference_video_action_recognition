import numpy as np
import time
import glob
import os

from flask import *

from python_src.opts import get_parser
from python_src.I3D import I3D
from python_src.InferenceModel import InferenceModel

app = Flask(__name__)

args = get_parser().parse_args()

# 推論モデルを定義
cls_model = InferenceModel(model=I3D(),
                           weight_path='./data/weights/rgb_imagenet.pkl',
                           label_map_path='./data/label_map.txt',
                           use_device=args.device)

# indexにリダイレクト
@app.route("/")
def redirectToIndex():
    return redirect(url_for('renderIndex'))

@app.route("/index", methods=['GET', 'POST'])
def renderIndex():
    video_list = [video_path.split('/')[-1] for video_path in glob.glob(os.path.join(args.sample_video_dir,'*.avi'))]
    return render_template("index.html",
                           video_list = video_list
                          )

@app.route("/result", methods=['POST'])
def renderResult():
    # プルダウンメニューで選択した動画のファイルパスを取得
    video_path = os.path.join(args.sample_video_dir,request.form.get('select_video'))

    # 動画ファイルの前処理と推論
    tic = time.time()
    video = cls_model.preprocessVideo(video_path=video_path)
    pred, logits = cls_model.inference(video)
    toc = time.time()
    
    # 結果を表示させるための文字列をprint_results_strに格納してレンダリング
    sorted_indices = np.argsort(pred)[::-1]
    print_results_str = []
    for index in sorted_indices[:10]:
        print_results_str.append('{}\t{} %'.format(cls_model.classes[index], round(pred[index]*100.0,3)))
    
    return render_template("result.html",
                           best_result = cls_model.classes[sorted_indices[0]],
                           time = round(toc-tic,3),
                           results = print_results_str
                          )

if __name__ == "__main__":
    app.run(debug=True, host=args.host, port=args.port, threaded=True)