import numpy as np
import time

from flask import *

from python_src.opts import get_parser
from python_src.I3D import I3D
from python_src.InferenceModel import InferenceModel

app = Flask(__name__)
cls_model = InferenceModel(model=I3D(),
                           weight_path='./data/weights/rgb_imagenet.pkl',
                           label_map_path='./data/label_map.txt',
                           use_device='cuda:0')

@app.route("/")
def redirectToIndex():
    return redirect(url_for('renderIndex'))

@app.route("/index", methods=['GET', 'POST'])
def renderIndex():
    return render_template("index.html")

@app.route("/result", methods=['POST'])
def renderResult():
    tic = time.time()
    video = cls_model.preprocessVideo(video_path='./sample_video/v_Basketball_g01_c01.avi')
    pred, logits = cls_model.inference(video)
    toc = time.time()
    
    sorted_indices = np.argsort(pred)[::-1]
    print_results_str = []
    for index in sorted_indices[:10]:
        print_results_str.append('{} {}'.format(cls_model.classes[index], round(pred[index]*100.0,3)))
    
    return render_template("result.html",
                           best_result = cls_model.classes[sorted_indices[0]],
                           time = round(toc-tic,3),
                           results = print_results_str
                          )


if __name__ == "__main__":
    args = get_parser().parse_args()
    app.run(debug=True, host=args.host, port=args.port, threaded=True)