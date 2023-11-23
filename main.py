from modal import Image, Mount, Stub, wsgi_app

stub = Stub()
image = (
    Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "wget", "git")
    .pip_install(
        "torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu"
    )
    .pip_install(
        "flask", "opencv-python", "matplotlib", "scikit-image", "pandas", "requests"
    )
)


@stub.function(image=image, mounts=[Mount.from_local_python_packages("yolo_backend")])
@wsgi_app()
def flask_app():
    import base64
    import time
    from pathlib import Path
    from pprint import pprint

    import numpy as np
    from flask import Flask, jsonify, request, send_from_directory

    import yolo_backend

    # Create necessary directories
    DATA_PATH = Path("data")
    to_create = [str(DATA_PATH), "data/to_prc", "data/gen_img"]
    for fpath in to_create:
        if not (p := Path(fpath)).exists():
            p.mkdir()

    # Flask app
    app = Flask(__name__)

    @app.route("/ping", methods=["GET"])
    def ping():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict_img():
        file = request.data
        cur_id = (
            str(int(time.time())) + "_" + str(np.random.randint(0, int(1e10))).zfill(10)
        )

        with open(DATA_PATH / f"to_prc/{cur_id}.jpg", "wb") as f:
            f.write(base64.b64decode(file))
        result = yolo_backend.predict_and_draw(
            DATA_PATH / "to_prc" / f"{cur_id}.jpg",
            DATA_PATH / "gen_img" / f"{cur_id}.jpg",
        ) | {"gen_img": f"gen_img/{cur_id}.jpg"}

        pprint(result)
        return jsonify(result)

    @app.route("/gen_img/<path:filepath>")
    def gen_img(filepath):
        return send_from_directory(DATA_PATH / "gen_img", filepath)

    return app
