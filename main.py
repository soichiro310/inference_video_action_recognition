import torch
import torch.nn as nn

from flask import *
from opts import get_parser
from python_src.i3d import InceptionI3d

app = Flask(__name__)

model = InceptionI3d(400, in_channels=3)
model.load_state_dict(torch.load('./model/rgb_imagenet.pt'))

@app.route("/")
def redirectToIndex():
    return redirect(url_for('renderIndex'))

@app.route("/index", methods=['GET', 'POST'])
def renderIndex():
    return render_template("index.html")

@app.route("/result", methods=['POST'])
def renderResult():
    return render_template("result.html")


if __name__ == "__main__":
    args = get_parser().parse_args()
    app.run(debug=True, host=args.host, port=args.port, threaded=True)