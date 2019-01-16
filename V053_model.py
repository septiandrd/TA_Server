import numpy as np
import tensorflow as tf
import scipy.misc

FLAGS = tf.flags.FLAGS

class Model:
    
    def __init__(self, name, features):
        self.name = name
        self.outputs = [features]

    def _get_layer_str(self, layer=None):
        if layer is None:
            layer = self.get_num_layers()
            return '%s_L%03d' % (self.name, layer+1)
        else :
            return '%s_%s' % (self.name, layer+str(self.get_num_layers()+1))

    def _get_num_inputs(self):
        return int(self.get_output().get_shape()[-1])

    def _glorot_initializer(self, prev_units, num_units, stddev_factor=2.0):
        stddev  = np.sqrt(stddev_factor / np.sqrt(prev_units*num_units))
        return tf.truncated_normal([prev_units, num_units],
                                    mean=0.0, stddev=stddev)

    def _glorot_initializer_conv2d(self, prev_units, num_units, mapsize, stddev_factor=2.0):
        stddev  = np.sqrt(stddev_factor / (np.sqrt(prev_units*num_units)*mapsize*mapsize))
        return tf.truncated_normal([mapsize, mapsize, prev_units, num_units],
                                    mean=0.0, stddev=stddev)

    def get_num_layers(self):
        return len(self.outputs)

    def add_batch_norm(self, scale=False):
        with tf.variable_scope(self._get_layer_str() ):
            out = tf.layers.batch_normalization(self.get_output(), scale=scale)
        
        self.outputs.append(out)
        return self

    def add_flatten(self):
        with tf.variable_scope(self._get_layer_str()):
            batch_size = int(self.get_output().get_shape()[0])
            out = tf.reshape(self.get_output(), [batch_size, -1])

        self.outputs.append(out)
        return self

    def add_dense(self, num_units, stddev_factor=1.0):
        assert len(self.get_output().get_shape()) == 2, "Previous layer must be 2-dimensional (batch, channels)"

        with tf.variable_scope(self._get_layer_str()):
            prev_units = self._get_num_inputs()
            
            # Weight term
            initw   = self._glorot_initializer(prev_units, num_units,
                                               stddev_factor=stddev_factor)
            weight  = tf.get_variable('weight', initializer=initw)

            # Bias term
            initb   = tf.constant(0.0, shape=[num_units])
            bias    = tf.get_variable('bias', initializer=initb)

            # Output of this layer
            out     = tf.matmul(self.get_output(), weight) + bias

        self.outputs.append(out)
        return self

    def add_sigmoid(self):
        with tf.variable_scope(self._get_layer_str()):
            # prev_units = self._get_num_inputs()
            out = tf.nn.sigmoid(self.get_output())
        
        self.outputs.append(out)
        return self

    def add_tanh(self):
        with tf.variable_scope(self._get_layer_str()):
            # prev_units = self._get_num_inputs()
            out = tf.nn.tanh(self.get_output())

        self.outputs.append(out)
        return self

    def add_softmax(self):
        with tf.variable_scope(self._get_layer_str()):
            this_input = tf.square(self.get_output())
            reduction_indices = list(range(1, len(this_input.get_shape())))
            acc = tf.reduce_sum(this_input, reduction_indices=reduction_indices, keep_dims=True)
            out = this_input / (acc+FLAGS.epsilon)
            #out = tf.verify_tensor_all_finite(out, "add_softmax failed; is sum equal to zero?")
        
        self.outputs.append(out)
        return self

    def add_relu(self):
        with tf.variable_scope(self._get_layer_str()):
            out = tf.nn.relu(self.get_output())

        self.outputs.append(out)
        return self        

    def add_elu(self):
        with tf.variable_scope(self._get_layer_str()):
            out = tf.nn.elu(self.get_output())

        self.outputs.append(out)
        return self

    def add_lrelu(self, leak=.2):
        with tf.variable_scope(self._get_layer_str()):
            t1  = .5 * (1 + leak)
            t2  = .5 * (1 - leak)
            out = t1 * self.get_output() + \
                  t2 * tf.abs(self.get_output())
            
        self.outputs.append(out)
        return self

    def add_dropout(self,keep_prob=0.4):
        with tf.variable_scope(self._get_layer_str()):
            out = tf.nn.dropout(self.get_output(),keep_prob=keep_prob)

        self.outputs.append(out)
        return self

    def add_conv2d(self, num_units, mapsize=1, stride=1, stddev_factor=2.0):
        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"
        
        with tf.variable_scope(self._get_layer_str(layer='conv2d') ):
            prev_units = self._get_num_inputs()
            
            initw  = self._glorot_initializer_conv2d(prev_units, num_units,
                                                 mapsize,
                                                 stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)
            out    = tf.nn.conv2d(self.get_output(), weight,
                                  strides=[1, stride, stride, 1],
                                  padding='SAME')

            initb  = tf.constant(0.0, shape=[num_units])
            bias   = tf.get_variable('bias', initializer=initb)
            out    = tf.nn.bias_add(out, bias)
            
        self.outputs.append(out)
        return self

    def conv2d(self, num_units, output, mapsize=1, stride=1, stddev_factor=2.0):
        assert len(output.get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        with tf.variable_scope(self._get_layer_str(layer='conv2d') ):
            # prev_units = self._get_num_inputs()
            prev_units = int(output.get_shape()[-1])

            initw = self._glorot_initializer_conv2d(prev_units, num_units,
                                                    mapsize,
                                                    stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)
            out = tf.nn.conv2d(output, weight,
                               strides=[1, stride, stride, 1],
                               padding='SAME')

            initb = tf.constant(0.0, shape=[num_units])
            bias = tf.get_variable('bias', initializer=initb)
            out = tf.nn.bias_add(out, bias)
        self.outputs.append(out)
        return out

    def add_conv2d_transpose(self, num_units, mapsize=1, stride=1, stddev_factor=2.0):
        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, " \
                                                           "height, channels) "

        with tf.variable_scope(self._get_layer_str('conv2d_T')):
            prev_units = self._get_num_inputs()
            
            # Weight term and convolution
            initw  = self._glorot_initializer_conv2d(prev_units, num_units,
                                                     mapsize,
                                                     stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)
            weight = tf.transpose(weight, perm=[0, 1, 3, 2])
            prev_output = self.get_output()
            output_shape = [FLAGS.batch_size,
                            int(prev_output.get_shape()[1]) * stride,
                            int(prev_output.get_shape()[2]) * stride,
                            num_units]
            out    = tf.nn.conv2d_transpose(self.get_output(), weight,
                                            output_shape=output_shape,
                                            strides=[1, stride, stride, 1],
                                            padding='SAME')

            # Bias term
            initb  = tf.constant(0.0, shape=[num_units])
            bias   = tf.get_variable('bias', initializer=initb)
            out    = tf.nn.bias_add(out, bias)
            
        self.outputs.append(out)
        return self

    def conv2d_transpose(self, num_units, output, mapsize=1, stride=1, stddev_factor=2.0):
        assert len(output.get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels) "

        with tf.variable_scope(self._get_layer_str('conv2d_T') ):
            # prev_units = self._get_num_inputs()
            prev_units = int(output.get_shape()[-1])

            initw = self._glorot_initializer_conv2d(prev_units, num_units,
                                                    mapsize,
                                                    stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)
            weight = tf.transpose(weight, perm=[0, 1, 3, 2])
            prev_output = output
            output_shape = [FLAGS.batch_size,
                            int(prev_output.get_shape()[1]) * stride,
                            int(prev_output.get_shape()[2]) * stride,
                            num_units]
            out = tf.nn.conv2d_transpose(output, weight,
                                         output_shape=output_shape,
                                         strides=[1, stride, stride, 1],
                                         padding='SAME')

            initb = tf.constant(0.0, shape=[num_units])
            bias = tf.get_variable('bias', initializer=initb)
            out = tf.nn.bias_add(out, bias)

        self.outputs.append(out)
        return out

    def add_inception_block(self, num_units, stddev_factor=2.0):
        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels) "

        prev_output = self.get_output()

        with tf.variable_scope(self._get_layer_str('inception_block')):
            out1 = self.conv2d(num_units, output=prev_output, mapsize=1, stride=1, stddev_factor=stddev_factor)
            out1 = tf.layers.batch_normalization(out1, scale=False)
            out1 = tf.nn.relu(out1)

            out2 = self.conv2d(num_units, output=prev_output, mapsize=3, stride=1, stddev_factor=stddev_factor)
            out2 = tf.layers.batch_normalization(out2, scale=False)
            out2 = tf.nn.relu(out2)

            out = tf.concat((out1, out2),axis=3)
            self.outputs.append(out)

        return self

    def add_front_block(self, num_units, stddev_factor=2.0) :
        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels) "

        prev_output = self.get_output()

        with tf.variable_scope(self._get_layer_str('front_block') ):
            out1 = self.conv2d(num_units=num_units, output=prev_output, mapsize=3, stride=1, stddev_factor=stddev_factor)
            out2 = self.conv2d(num_units=num_units, output=prev_output, mapsize=5, stride=1, stddev_factor=stddev_factor)

            cat = tf.concat((out1,out2),axis=3)

            # is it better to use batchnorm ?
            out = tf.layers.batch_normalization(cat, scale=False)
            out = tf.nn.relu(out)

            self.outputs.append(out)

        return self
        
    def add_inception_residual_block(self, num_units, stddev_factor=2.0):
        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels) "
        
        prev_output = self.get_output()

        with tf.variable_scope(self._get_layer_str('inception_residual_block') ):
            out1 = self.conv2d(num_units=num_units, output=prev_output, mapsize=3, stride=1, stddev_factor=stddev_factor)
            out1 = tf.layers.batch_normalization(out1, scale=False)
            out1 = tf.nn.relu(out1)
            out1 = self.conv2d(num_units=num_units, output=out1, mapsize=3, stride=1, stddev_factor=stddev_factor)
        
            out2 = self.conv2d(num_units=num_units, output=prev_output, mapsize=5, stride=1, stddev_factor=stddev_factor)
            out2 = tf.layers.batch_normalization(out2, scale=False)
            out2 = tf.nn.relu(out2)
            out2 = self.conv2d(num_units=num_units, output=out2, mapsize=5, stride=1, stddev_factor=stddev_factor)

            cat = tf.concat((out1,out2),axis=3)

            add = tf.add(cat,prev_output)

            # is it better to use batchnorm ?
            out = tf.layers.batch_normalization(add, scale=False)
            out = tf.nn.relu(out)

            self.outputs.append(out)

        return self
        
    def add_upscaling_block(self, num_units, stddev_factor=2.0):
        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels) "
        
        prev_output = self.get_output()

        with tf.variable_scope(self._get_layer_str('upscaling_block') ):
            out1 = self.conv2d_transpose(num_units=num_units, output=prev_output, mapsize=3, stride=2, stddev_factor=stddev_factor)
            out2 = self.conv2d_transpose(num_units=num_units, output=prev_output, mapsize=6, stride=2, stddev_factor=stddev_factor)

            cat = tf.concat((out1,out2),axis=3)

            # is it better to use batchnorm ?
            out = tf.layers.batch_normalization(cat, scale=False)
            out = tf.nn.relu(out)

            self.outputs.append(out)

        return self


    def add_residual_block(self, num_units, mapsize=3, num_layers=2, stddev_factor=2.0):
        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        if num_units != int(self.get_output().get_shape()[3]):
            print(self._get_layer_str())
            self.add_conv2d(num_units, mapsize=1, stride=1, stddev_factor=2.0)

        bypass = self.get_output()

        for _ in range(num_layers):
            self.add_batch_norm()
            self.add_relu()
            self.add_conv2d(num_units, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)

        self.add_sum(bypass)

        return self

    def add_bottleneck_residual_block(self, num_units, mapsize=3, stride=1, transpose=False):
        """Adds a bottleneck residual block as per Arxiv 1512.03385, Figure 3"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        # Add projection in series if needed prior to shortcut
        if num_units != int(self.get_output().get_shape()[3]) or stride != 1:
            ms = 1 if stride == 1 else mapsize
            #bypass.add_batch_norm() # TBD: Needed?
            if transpose:
                self.add_conv2d_transpose(num_units, mapsize=ms, stride=stride, stddev_factor=1.)
            else:
                self.add_conv2d(num_units, mapsize=ms, stride=stride, stddev_factor=1.)

        bypass = self.get_output()

        # Bottleneck residual block
        self.add_batch_norm()
        self.add_relu()
        self.add_conv2d(num_units//4, mapsize=1,       stride=1,      stddev_factor=2.)

        self.add_batch_norm()
        self.add_relu()
        if transpose:
            self.add_conv2d_transpose(num_units//4,
                                      mapsize=mapsize,
                                      stride=1,
                                      stddev_factor=2.)
        else:
            self.add_conv2d(num_units//4,
                            mapsize=mapsize,
                            stride=1,
                            stddev_factor=2.)

        self.add_batch_norm()
        self.add_relu()
        self.add_conv2d(num_units,    mapsize=1,       stride=1,      stddev_factor=2.)

        self.add_sum(bypass)

        return self

    def add_sum(self, term):
        """Adds a layer that sums the top layer with the given term"""

        with tf.variable_scope(self._get_layer_str()):
            prev_shape = self.get_output().get_shape()
            term_shape = term.get_shape()
            #print("%s %s" % (prev_shape, term_shape))
            assert prev_shape == term_shape and "Can't sum terms with a different size"
            out = tf.add(self.get_output(), term)
        
        self.outputs.append(out)
        return self

    def add_mean(self):
        """Adds a layer that averages the inputs from the previous layer"""

        with tf.variable_scope(self._get_layer_str()):
            prev_shape = self.get_output().get_shape()
            reduction_indices = list(range(len(prev_shape)))
            assert len(reduction_indices) > 2 and "Can't average a (batch, activation) tensor"
            reduction_indices = reduction_indices[1:-1]
            out = tf.reduce_mean(self.get_output(), reduction_indices=reduction_indices)
        
        self.outputs.append(out)
        return self

    def add_upscale(self):
        """Adds a layer that upscales the output by 2x through nearest neighbor interpolation"""

        prev_shape = self.get_output().get_shape()
        size = [2 * int(s) for s in prev_shape[1:3]]
        out  = tf.image.resize_nearest_neighbor(self.get_output(), size)

        self.outputs.append(out)
        return self

    def upscale(self,output,K):
        """Adds a layer that upscales the output by 2x through nearest neighbor interpolation"""

        prev_shape = output.get_shape()
        size = [K * int(s) for s in prev_shape[1:3]]
        out  = tf.image.resize_nearest_neighbor(output, size)

        # self.outputs.append(out)
        return out

    def add_skip_connection(self):

        with tf.variable_scope(self._get_layer_str('skip_connection')):
            input = self.get_all_outputs()[0]
            print
            input = self.upscale(output=input, K=4)
            output = self.get_all_outputs()[-1]

            out = tf.add(input,output)
            self.outputs.append(out)
        return out

    def get_output(self):
        """Returns the output from the topmost layer of the network"""
        return self.outputs[-1]

    def get_all_outputs(self):
        return self.outputs

    def get_variable(self, layer, name):
        """Returns a variable given its layer and name.

        The variable must already exist."""

        scope      = self._get_layer_str(layer)
        collection = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)

        # TBD: Ugly!
        for var in collection:
            if var.name[:-2] == scope+'/'+name:
                return var

        return None

    def get_all_layer_variables(self, layer):
        """Returns all variables in the given layer"""
        scope = self._get_layer_str(layer)
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)

def _discriminator_model(sess, features, disc_input):
    mapsize = 3
    layers = [64, 128, 256, 512]

    old_vars = tf.all_variables()

    model = Model('DIS',disc_input)

    for layer in range(len(layers)):
        nunits = layers[layer]
        stddev_factor = 2.0

        model.add_conv2d(nunits, mapsize=mapsize, stride=2, stddev_factor=stddev_factor)
        model.add_batch_norm()
        model.add_relu()

    model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
    model.add_batch_norm()
    model.add_relu()

    model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=stddev_factor)
    model.add_batch_norm()
    model.add_relu()

    model.add_conv2d(1, mapsize=1, stride=1, stddev_factor=stddev_factor)
    model.add_mean()

    new_vars = tf.all_variables()
    disc_vars = list(set(new_vars) - set(old_vars))
    extracted_features = model.get_all_outputs()[-9]

    return model.get_output(), extracted_features, disc_vars

def _generator_model(sess, features, labels):
    old_vars = tf.all_variables()

    model = Model('GEN', features)

    model.add_front_block(num_units=256)
    model.add_inception_residual_block(num_units=256)
    model.add_inception_residual_block(num_units=256)
    model.add_upscaling_block(num_units=128)
    model.add_inception_residual_block(num_units=128)
    model.add_inception_residual_block(num_units=128)
    model.add_upscaling_block(num_units=64)
    model.add_conv2d(num_units=64, mapsize=3, stride=1)
    model.add_batch_norm()
    model.add_relu()
    model.add_conv2d(num_units=32, mapsize=3, stride=1)
    model.add_batch_norm()
    model.add_relu()
    model.add_conv2d(num_units=3, mapsize=1, stride=1)
    model.add_batch_norm()
    model.add_tanh()
    
    new_vars  = tf.all_variables()
    gene_vars = list(set(new_vars) - set(old_vars))

    return model.get_output(), gene_vars

def create_model(sess, features, labels):
    # Generator
    rows      = int(features.get_shape()[1])
    cols      = int(features.get_shape()[2])
    channels  = int(features.get_shape()[3])

    gene_minput = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, rows, cols, channels])

    # TBD: Is there a better way to instance the generator?
    with tf.variable_scope('gene') as scope:
        gene_output, gene_var_list = \
                    _generator_model(sess, features, channels)

        scope.reuse_variables()

        gene_moutput, _ = _generator_model(sess, gene_minput, channels)
    
    # Discriminator with real data
    disc_real_input = tf.identity(labels, name='disc_real_input')

    # TBD: Is there a better way to instance the discriminator?
    with tf.variable_scope('disc') as scope:
        disc_real_output, disc_real_feature, disc_var_list = \
                _discriminator_model(sess, features, disc_real_input)

        scope.reuse_variables()
            
        disc_fake_output, disc_fake_feature, _ = _discriminator_model(sess, features, gene_output)

    return [gene_minput,      gene_moutput,
            gene_output,      gene_var_list,
            disc_real_output, disc_real_feature, 
            disc_fake_output, disc_fake_feature, disc_var_list]

def _load_model(sess) :
    gene_minput = tf.placeholder(tf.float32, shape=[1, 16, 16, 3])
    with tf.variable_scope('gene') as scope:
        gene_moutput, _ = _generator_model(sess, gene_minput, 3)

    return gene_minput, gene_moutput

def _downscale(images, K):
    """Differentiable image downscaling by a factor of K"""
    arr = np.zeros([K, K, 3, 3])
    arr[:,:,0,0] = 1.0/(K*K)
    arr[:,:,1,1] = 1.0/(K*K)
    arr[:,:,2,2] = 1.0/(K*K)
    dowscale_weight = tf.constant(arr, dtype=tf.float32)
    
    downscaled = tf.nn.conv2d(images, dowscale_weight,
                              strides=[1, K, K, 1],
                              padding='SAME')
    return downscaled

def create_generator_loss(disc_output, disc_real_feature, disc_fake_feature, gene_output, labels):
    # I.e. did we fool the discriminator?
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_output, labels=tf.ones_like(disc_output))
    gene_ce_loss  = tf.reduce_mean(cross_entropy, name='gene_ce_loss')

    # I.e. does the result look like the feature?
    # K = int(gene_output.get_shape()[1])//int(features.get_shape()[1])
    # assert K == 2 or K == 4 or K == 8    
    # downscaled = _downscale(gene_output, K)
    
    # gene_l1_loss  = tf.reduce_mean(tf.abs(gene_output - labels), name='gene_l1_loss')
    gene_l2_loss  = tf.norm(gene_output - labels, ord='euclidean', name='gene_l2_loss')
    gene_l2_feature_loss =  tf.norm(disc_fake_feature - disc_real_feature, ord='euclidean', name='gene_l2_loss')
    psnr_loss = tf.reduce_mean(tf.image.psnr(gene_output, labels, max_val=1.0), name='psnr_loss')

    gene_loss     = tf.add_n([0.2 * gene_ce_loss,
                           0.6 * gene_l2_loss,
                           0.2 * gene_l2_feature_loss,
                           -0.2 * psnr_loss], name='gene_loss')
    
    return gene_loss

# def calculate_psnr_score(disc_output, gene_output, labels):

#     generated_imgs = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)
#     x = scipy.misc.toimage(generated_imgs, cmin=0., cmax=1.)
#     y = scipy.misc.toimage(labels, cmin=0., cmax=1.)
#     y = scipy.misc.toimage
#     psnr_score = tf.image.psnr(generated_imgs, labels, max_val=1.0)

#     return psnr_score

def create_discriminator_loss(disc_real_output, disc_fake_output):
    # I.e. did we correctly identify the input as real or not?
    cross_entropy_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_output, labels=tf.ones_like(disc_real_output))
    disc_real_loss     = tf.reduce_mean(cross_entropy_real, name='disc_real_loss')
    
    cross_entropy_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_output, labels=tf.zeros_like(disc_fake_output))
    disc_fake_loss     = tf.reduce_mean(cross_entropy_fake, name='disc_fake_loss')

    return disc_real_loss, disc_fake_loss

def create_optimizers(gene_loss, gene_var_list,
                      disc_loss, disc_var_list):    
    # TBD: Does this global step variable need to be manually incremented? I think so.
    global_step    = tf.Variable(0, dtype=tf.int64,   trainable=False, name='global_step')
    learning_rate  = tf.placeholder(dtype=tf.float32, name='learning_rate')
    
    gene_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=FLAGS.learning_beta1,
                                       name='gene_optimizer')
    disc_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=FLAGS.learning_beta1,
                                       name='disc_optimizer')

    gene_minimize = gene_opti.minimize(gene_loss, var_list=gene_var_list, name='gene_loss_minimize', global_step=global_step)
    
    disc_minimize     = disc_opti.minimize(disc_loss, var_list=disc_var_list, name='disc_loss_minimize', global_step=global_step)
    
    return (global_step, learning_rate, gene_minimize, disc_minimize)
