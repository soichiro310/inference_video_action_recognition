import numpy as np
import time
import glob
import os
import random
import string

from flask import *

from python_src.opts import get_parser
from python_src.I3D import I3D
from python_src.InferenceModel import InferenceModel
from python_src.myException import *

def createApp(args):
    # Flaskクラスのインスタンス生成
    app = Flask(__name__)

    # シークレットキー生成
    app.secret_key = "".join([random.choice(string.ascii_letters + string.digits + '_' + '-' + '!' + '#' + '&') for i in range(64)])
    
    # 推論モデルを定義
    cls_model = InferenceModel(model=I3D(),
                            weight_path='./data/weights/rgb_imagenet.pkl',
                            label_map_path='./data/label_map.txt',
                            )
    
    # indexにリダイレクト
    @app.route("/")
    def redirectToIndex():
        return redirect(url_for('renderIndex'))

    @app.route("/index", methods=['GET', 'POST'])
    def renderIndex():
        # コマンドライン引数で指定したディレクトリ内にある動画ファイルのファイルパスに関するリストを作成
        video_list = [video_path.split('/')[-1] for video_path in glob.glob(os.path.join(args.sample_video_dir,'*.avi'))]

        # 選択したファイルに問題があった場合はindex.htmlにエラーメッセージを表示させる
        error = session['error'] if 'error' in session else ''

        return render_template("index.html",
                            error = error,
                            video_list = video_list
                            )

    @app.route('/post', methods=['POST'])
    def dataCheck():
        try:
            # プルダウンメニューで選択した動画のファイルパスを取得
            video_name = request.form.get('select_video')
            video_path = os.path.join(args.sample_video_dir,video_name)

            # 動画ファイルの推論
            tic = time.time()
            pred = cls_model.inferenceVideo(video_path=video_path)
            toc = time.time()
        except TypeError:   
            # プルダウンの初期値のまま，ボタンを押してしまうと起きてしまう例外
            session['error'] = '動画ファイル(.avi)を選択してください．'
            return redirect(url_for('renderIndex'))
        except VideoOpenError:  
            # ファイルが開けなかった場合，エラーメッセージを表示させる処理を行い，indexにリダイレクト
            session['error'] = '選択したファイル('+ video_name +')が開けませんでした．'
            return redirect(url_for('renderIndex'))

        # 結果を表示させるための文字列をprint_results_strに格納してレンダリング
        sorted_indices = np.argsort(pred)[::-1]
        print_results_str = []
        for index in sorted_indices[:10]:
            print_results_str.append('{}\t{} %'.format(cls_model.classes[index], round(pred[index]*100.0,3)))

        session['error'] = ''   # 問題なく推論が行われた場合，indexに表示させるエラーメッセージをクリア
        session['video'] = request.form['select_video']
        session['best_result'] = cls_model.classes[sorted_indices[0]]
        session['time'] = round(toc-tic,3)
        session['results'] = print_results_str

        return redirect(url_for('renderResult'))

    @app.route("/result", methods=['GET', 'POST'])
    def renderResult():
        if 'best_result' not in session:
            # 推論結果を格納するセッション変数が存在しない場合，indexにリダイレクト
            return redirect(url_for('renderIndex'))
        else :
            return render_template("result.html",
                                best_result = session['best_result'],
                                time = session['time'],
                                results = session['results']
                                )

    return app

if __name__ == "__main__":
    args = get_parser().parse_args()
    app = createApp(args)
    app.run(debug=args.debug, host=args.host, port=args.port, threaded=True)
