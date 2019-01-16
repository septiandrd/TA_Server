import V053_model as model
import V053_input as input

from keras.models import load_model

import os.path
import random
import tensorflow as tf
import scipy.misc
import numpy as np
import cv2

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 1,
                            "Number of samples per batch.")

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint_4',
                           "Output folder where checkpoints are dumped.")

tf.app.flags.DEFINE_integer('test_vectors', 1,
                            """Number of features to use for testing""")

tf.app.flags.DEFINE_integer('sample_size', 64,
                            "Image sample size in pixels. Range [64,128]")

tf.app.flags.DEFINE_string('train_dir', 'train',
                           "Output folder where training logs are dumped.")

tf.app.flags.DEFINE_float('learning_beta1', 0.5,
                          "Beta1 parameter used for AdamOptimizer")


def load_FRmodel():
    FRmodel = load_model('recognition_2.h5', compile=False)
    return FRmodel


def img_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)

    img2 = cv2.resize(img1, (96, 96))
    img = img2[..., ::-1]

    img = np.transpose(img, (2, 0, 1))
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding


def encoding_distance(input_path, output_path, FRmodel):
    # FRmodel = load_model('recognition_2.h5', compile=False)
    # print(input_path)
    # print(output_path)
    a = img_to_encoding(input_path, FRmodel)
    b = img_to_encoding(output_path, FRmodel)
    dist = np.linalg.norm(a - b)

    return dist


def setup_tensorflow():
    config = tf.ConfigProto(log_device_placement=False)
    sess = tf.Session(config=config)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    return sess, summary_writer


class TrainData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


def test_model(train_data):

    newname = 'checkpoint_new.txt'
    newname = os.path.join(FLAGS.checkpoint_dir, newname)
    td = train_data

    saver = tf.train.Saver()
    saver.restore(td.sess, newname)
    print("RESTORED")

    test_feature, test_label = td.sess.run([td.test_features, td.test_labels])

    feed_dict = {td.gene_minput: test_feature}
    gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dict)

    size = [64, 64]

    input_imgs = tf.image.resize_nearest_neighbor(test_feature, size)
    input_imgs = tf.maximum(tf.minimum(input_imgs, 1.0), 0.0)
    generated_imgs = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)

    # bicubic = tf.image.resize_bicubic(test_feature, size)
    # bicubic = tf.maximum(tf.minimum(bicubic, 1.0), 0.0)

    # psnr_score_n = tf.reduce_mean(tf.image.psnr(test_label, input_imgs, max_val=1.0))
    # psnr_score_n = td.sess.run(psnr_score_n)
    # print(psnr_score_n)
    # ssim_score_n = tf.reduce_mean(tf.image.ssim(tf.maximum(tf.minimum(test_label, 1.0), 0.0), input_imgs, max_val=1.0))
    # ssim_score_n = td.sess.run(ssim_score_n)
    # print(ssim_score_n)
    # psnr_score_b = tf.reduce_mean(tf.image.psnr(test_label, bicubic, max_val=1.0))
    # psnr_score_b = td.sess.run(psnr_score_b)
    # print(psnr_score_b)
    # ssim_score_b = tf.reduce_mean(tf.image.ssim(tf.maximum(tf.minimum(test_label, 1.0), 0.0), bicubic, max_val=1.0))
    # ssim_score_b = td.sess.run(ssim_score_b)
    # print(ssim_score_b)
    # psnr_score_g = tf.reduce_mean(tf.image.psnr(test_label, gene_output, max_val=1.0))
    # psnr_score_g = td.sess.run(psnr_score_g)
    # print(psnr_score_g)
    # ssim_score_g = tf.reduce_mean(tf.image.ssim(tf.maximum(tf.minimum(test_label, 1.0), 0.0), generated_imgs, max_val=1.0))
    # ssim_score_g = td.sess.run(ssim_score_g)
    # print(ssim_score_g)

    output_filenames = []

    images = td.sess.run(input_imgs)
    for img in images:
        filename = td.all_filenames[0].split(
            '/')[2].split('.')[-2]+'-input.jpg'
        output_filenames.append(filename)
        filename = os.path.join('storage', filename)
        scipy.misc.toimage(img, cmin=0., cmax=1.).save(filename)
        print("Image saved.")

    images = td.sess.run(generated_imgs)
    for img in images:
        filename = td.all_filenames[0].split(
            '/')[2].split('.')[-2]+'-output.jpg'
        output_filenames.append(filename)
        filename = os.path.join('storage', filename)
        scipy.misc.toimage(img, cmin=0., cmax=1.).save(filename)
        print("Image saved.")

    for img in test_label:
        filename = td.all_filenames[0].split(
            '/')[2].split('.')[-2]+'-label.jpg'
        output_filenames.append(filename)
        filename = os.path.join('storage', filename)
        scipy.misc.toimage(img, cmin=0., cmax=1.).save(filename)
        print("Image saved.")

    return output_filenames


def _run(filename):
    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    all_filenames = [filename]

    img_count = len(all_filenames)

    train_filenames = all_filenames
    test_filenames = all_filenames

    train_features, train_labels = input.setup_inputs(sess, train_filenames)
    test_features, test_labels = input.setup_inputs(sess, test_filenames)

    # Add some noise during training (think denoising autoencoders)
    noise_level = .02
    noisy_train_features = train_features + \
        tf.random_normal(train_features.get_shape(), stddev=noise_level)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_real_feature,
     disc_fake_output, disc_fake_feature, disc_var_list] = \
        model.create_model(sess, noisy_train_features, train_labels)

    gene_loss = model.create_generator_loss(
        disc_fake_output, disc_real_feature, disc_fake_feature, gene_output, train_labels)

    disc_real_loss, disc_fake_loss = \
        model.create_discriminator_loss(disc_real_output, disc_fake_output)
    disc_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')

    (global_step, learning_rate, gene_minimize, disc_minimize) = \
        model.create_optimizers(gene_loss, gene_var_list,
                                disc_loss, disc_var_list)

    writer = tf.summary.FileWriter('./tensorboard', sess.graph)

    # Train model
    train_data = TrainData(locals())
    output_filenames = test_model(train_data)

    tf.reset_default_graph()
    return output_filenames


# def load(sess):
#     [gene_minput, gene_moutput] = model._load_model(sess)

#     return gene_minput, gene_moutput


def predict(graph, sess, gene_minput, gene_moutput, input_filename):
    with graph.as_default():
        test_features, test_labels = input.setup_inputs(sess, [input_filename])
        test_feature, test_label = sess.run([test_features, test_labels])

        feed_dict = {gene_minput: test_feature}
        gene_output = sess.run(gene_moutput, feed_dict=feed_dict)

        size = [64, 64]

        input_imgs = tf.image.resize_nearest_neighbor(test_feature, size)
        input_imgs = tf.maximum(tf.minimum(input_imgs, 1.0), 0.0)
        generated_imgs = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)

        # bicubic = tf.image.resize_bicubic(test_feature, size)
        # bicubic = tf.maximum(tf.minimum(bicubic, 1.0), 0.0)

        # psnr_score_n = tf.reduce_mean(tf.image.psnr(test_label, input_imgs, max_val=1.0))
        # psnr_score_n = td.sess.run(psnr_score_n)
        # print(psnr_score_n)
        # ssim_score_n = tf.reduce_mean(tf.image.ssim(tf.maximum(tf.minimum(test_label, 1.0), 0.0), input_imgs, max_val=1.0))
        # ssim_score_n = td.sess.run(ssim_score_n)
        # print(ssim_score_n)
        # psnr_score_b = tf.reduce_mean(tf.image.psnr(test_label, bicubic, max_val=1.0))
        # psnr_score_b = td.sess.run(psnr_score_b)
        # print(psnr_score_b)
        # ssim_score_b = tf.reduce_mean(tf.image.ssim(tf.maximum(tf.minimum(test_label, 1.0), 0.0), bicubic, max_val=1.0))
        # ssim_score_b = td.sess.run(ssim_score_b)
        # print(ssim_score_b)
        # psnr_score_g = tf.reduce_mean(tf.image.psnr(test_label, gene_output, max_val=1.0))
        # psnr_score_g = td.sess.run(psnr_score_g)
        # print(psnr_score_g)
        # ssim_score_g = tf.reduce_mean(tf.image.ssim(tf.maximum(tf.minimum(test_label, 1.0), 0.0), generated_imgs, max_val=1.0))
        # ssim_score_g = td.sess.run(ssim_score_g)
        # print(ssim_score_g)

        output_filenames = []

        images = sess.run(input_imgs)
        for img in images:
            filename = input_filename.split(
                '/')[2].split('.')[-2]+'-input.jpg'
            output_filenames.append(filename)
            filename = os.path.join('storage', filename)
            scipy.misc.toimage(img, cmin=0., cmax=1.).save(filename)
            print("Image saved.")

        images = sess.run(generated_imgs)
        for img in images:
            filename = input_filename.split(
                '/')[2].split('.')[-2]+'-output.jpg'
            output_filenames.append(filename)
            filename = os.path.join('storage', filename)
            scipy.misc.toimage(img, cmin=0., cmax=1.).save(filename)
            print("Image saved.")

        for img in test_label:
            filename = input_filename.split(
                '/')[2].split('.')[-2]+'-label.jpg'
            output_filenames.append(filename)
            filename = os.path.join('storage', filename)
            scipy.misc.toimage(img, cmin=0., cmax=1.).save(filename)
            print("Image saved.")

        return output_filenames


def load(filename='./storage/0a32432d-bbdf-4c12-be6e-3bfcc86d14ad-cropped.jpg'):
    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    all_filenames = [filename]

    img_count = len(all_filenames)

    train_filenames = all_filenames
    test_filenames = all_filenames

    train_features, train_labels = input.setup_inputs(sess, train_filenames)
    test_features, test_labels = input.setup_inputs(sess, test_filenames)

    # Add some noise during training (think denoising autoencoders)
    noise_level = .02
    noisy_train_features = train_features + \
        tf.random_normal(train_features.get_shape(), stddev=noise_level)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_real_feature,
     disc_fake_output, disc_fake_feature, disc_var_list] = \
        model.create_model(sess, noisy_train_features, train_labels)

    gene_loss = model.create_generator_loss(
        disc_fake_output, disc_real_feature, disc_fake_feature, gene_output, train_labels)

    disc_real_loss, disc_fake_loss = \
        model.create_discriminator_loss(disc_real_output, disc_fake_output)
    disc_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')

    (global_step, learning_rate, gene_minimize, disc_minimize) = \
        model.create_optimizers(gene_loss, gene_var_list,
                                disc_loss, disc_var_list)

    writer = tf.summary.FileWriter('./tensorboard', sess.graph)

    # Train model
    # train_data = TrainData(locals())
    # output_filenames = test_model(train_data)

    newname = 'checkpoint_new.txt'
    newname = os.path.join(FLAGS.checkpoint_dir, newname)
    # td = train_data

    saver = tf.train.Saver()
    saver.restore(sess, newname)

    graph = tf.get_default_graph()

    return graph, sess, gene_minput, gene_moutput
