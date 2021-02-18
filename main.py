from flask import *
from opts import get_parser

app = Flask(__name__)

@app.route("/")
def main():
    return "Hello World"


if __name__ == "__main__":
    args = get_parser().parse_args()
    app.run(debug=True, host=args.host, port=args.port, threaded=True)