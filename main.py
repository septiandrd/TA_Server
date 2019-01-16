from flask import Flask, request, json, send_file
from flask_cors import CORS
import uuid
import cv2

import preprocessing
import testing
import tensorflow as tf

# from keras.models import load_model

app = Flask(__name__)
CORS(app)

# global graph, sess, gene_minput, gene_moutput
# # sess = tf.Session()
# # saver = tf.train.import_meta_graph('./checkpoint_4/checkpoint_new.txt.meta')
# graph, sess, gene_minput, gene_moutput = testing.load()
# # saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_4/'))
# print("YOI")


@app.route("/")
def index():
    return "Hello World!"


@app.route("/degrade-reconstruct", methods=['POST'])
def degrade_reconstruct():
    img_input = request.files['img_input']

    filename = str(uuid.uuid4()) + ".jpg"

    ori_path = './storage/'+filename

    img_input.save(ori_path)

    cropped_path = preprocessing.crop('./storage/'+filename)

    # outputs = testing.predict(graph, sess, gene_minput, gene_minput, cropped_path)

    outputs = testing._run(cropped_path)

    input_path = './storage/'+outputs[0]
    output_path = './storage/'+outputs[1]
    label_path = './storage/'+outputs[2]

    return json.dumps({'input_filename': outputs[0], 'output_filename': outputs[1], 'label_filename': outputs[2]})


@app.route('/get_image')
def get_image():
    path = './storage/'+request.args.get('filename')
    return send_file(path, mimetype='image/gif')


if __name__ == "__main__":
    app.run(debug=True, port=1234)
