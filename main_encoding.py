from flask import Flask, request, json
from flask_cors import CORS

import numpy as np
from keras.models import load_model
import tensorflow as tf
import cv2

app = Flask(__name__)
CORS(app)

global FRmodel
FRmodel = load_model('recognition_2.h5', compile=False)
global graph
graph = tf.get_default_graph()

@app.route("/encoding")
def degrade_reconstruct():
    input_path = './storage/'+request.args.get('input_filename')
    output_path = './storage/'+request.args.get('output_filename')
    label_path = './storage/'+request.args.get('label_filename')

    distance_generated = encoding_distance(label_path, output_path, FRmodel)
    distance_input = encoding_distance(label_path, input_path, FRmodel)

    print("distance input = "+str(distance_input))
    print("distance generated = "+str(distance_generated))

    return json.dumps({'distance_input': str(distance_input), 'distance_generated': str(distance_generated)})

def img_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)

    img2 = cv2.resize(img1, (96, 96))
    img = img2[..., ::-1]

    img = np.transpose(img, (2, 0, 1))
    x_train = np.array([img])
    with graph.as_default():
      embedding = model.predict_on_batch(x_train)
    return embedding

def encoding_distance(input_path, output_path, FRmodel):
    a = img_to_encoding(input_path, FRmodel)
    b = img_to_encoding(output_path, FRmodel)
    dist = np.linalg.norm(a - b)

    return dist

if __name__ == "__main__":
    app.run(debug=True,port=4321)