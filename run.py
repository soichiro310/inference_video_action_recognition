import torch
import torch.nn as nn

from flask import *
from python_src.opts import get_parser
from python_src.I3D import I3D
from python_src.InferenceModel import InferenceModel

app = Flask(__name__)

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