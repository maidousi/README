from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import os
import lvm_read
# import cPickle as pickle
import pickle
from os.path import exists
import pandas as pd
#把train_data取出来

def scale_data(train_data, test_data):

    data = train_data.reshape((-1, train_data.shape[2]))
    scaler = StandardScaler()
    scaler.fit(data)

    # Scale the training data
    for i in range(train_data.shape[0]):
        train_data[i] = scaler.transform(train_data[i])

    # Scale the testing data
    for i in range(test_data.shape[0]):
        test_data[i] = scaler.transform(test_data[i])

    return train_data, test_data
def extract_data(train_path='../neural_bof/splitknifedata/train', test_path='../neural_bof/splitknifedata/test'):
    """
    Transforms the EEG dataset into the appropriate format
    :param train_path:
    :param test_path:
    :return: 需要重写
    """

    def load_from_folder(path, threshold):
        files = [os.path.join(path, x) for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]
        eeg_data = []
        eeg_label = []
        # run_threshold =
        for file_path in files:
            df = pd.read_csv(file_path)
            data = df['values'].tolist()
            file_prefix = os.path.split(file_path)[-1].split('_')[2]
            # data = [x for i, x in enumerate(data) if x > run_threshold]
            data = [x for i, x in enumerate(data)]
            # data1 = data + [0] * (100 - len(data))
            values = np.float64(data).reshape((1, 100))

            if file_prefix < threshold:
                label = 0
            else:
                label = 1

            if label == 1:
                eeg_data.append(values)
                eeg_label.append(label)
        eeg_data = np.float64(eeg_data)

        return eeg_data, eeg_label


    print("Extracting dataset...")
    data = []
    labels = []
    file_dict = {'201804031620-201804041053': '1054', '201804040814-201804040901': '0902',
    '201804040909-201804040956': '0957', '201804041004-201804041051': '1040',
    '201803071408-201803071541': '1523', '201803071549-201803071737': '1710',
    '201803081357-201803081458': '1449', '201803081515-201803081610': '1601',
    '201803090903-201803091003': '0956', '201803091054-201803091146': '1140',
    '201803091450-201803091612': '1600', '201803080827-201803080948': '0949'
    }

    for folder in os.listdir(train_path):
        print(os.path.join(train_path, folder))
        if os.path.isdir(os.path.join(train_path, folder)):
            threshold = file_dict[folder]
            datas, label = load_from_folder(os.path.join(train_path, folder), threshold)
            if not np.isnan(datas).all():
                data.append(datas)
                labels.append(label)
            print(len(data))
            print(len(labels))
    train_data = np.float64(np.concatenate(data))
    train_labels = np.float64(np.concatenate(labels))

    data = []
    labels = []
    for folder in os.listdir(test_path):
        if os.path.isdir(os.path.join(test_path, folder)):
            threshold = file_dict[folder]
            data_sg, label_sg = load_from_folder(os.path.join(test_path, folder), threshold)
            data.append(data_sg)
            labels.append(label_sg)
    test_data = np.float64(np.concatenate(data))
    test_labels = np.float64(np.concatenate(labels))

    with open('data/negative_data.pickle', 'wb') as f:
        pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_labels, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_labels, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_dataset():
    """
    Loads the EEG dataset
    :return:
    """

    if not exists('data/negative_data.pickle'):
        extract_data()

    with open('data/negative_data.pickle', 'rb') as f:
        train_data = pickle.load(f)
        train_labels = pickle.load(f)
        test_data = pickle.load(f)
        test_labels = pickle.load(f)

    return train_data, train_labels, test_data, test_labels

"""
if __name__ == "__main__":
    load_dataset()

"""
# unit = 200
# H = tf.layers.dense(input, units=unit, activation='relu',
#                 kernel_initialzier=tf.random_normal_initializer(stddev=0.1),
#                 # bias_initializer=tf.constant_initializer(0.0))
# H = tf.layers.batch_normalization(H)

def deconv2d(tensor, filters, ksize, stride):
    deconv = tf.layers.conv2d_transpose(tensor, filters, kernel_size=(ksize, 1), strides=(stride, 1),
                                        padding='valid',
                                        # name="conv1"
                                      )
    bn = tf.layers.batch_normalization(deconv)
    activation = tf.nn.relu(bn)
    return activation



# def generator(batch_size):
def generator(noise):
    # noise = tf.random_normal(shape=[batch_size, 128], dtype=tf.float32)
    # noise = tf.reshape(noise, [batch_size,128],'noise')

    unit = 512
    fc = tf.layers.dense(noise, units=unit,
                kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                # bias_initializer=tf.constant_initializer(0.0)
    )
    fc_bn = tf.layers.batch_normalization(fc)
    fc_out = tf.nn.relu(fc_bn)
    fc_out = tf.reshape(fc_out, [batch_size, unit, 1, 1])

    for i in range(2):
        fc_out = deconv2d(fc_out, filters=256, ksize=3, stride=1)

    deconv_out = tf.layers.conv2d_transpose(fc_out, filters=100, kernel_size=(3, 1), strides=(1, 1),
                                        padding='same',
                                        # name="conv1"
                                      )
    out = tf.nn.sigmoid(deconv_out)
    out = tf.reshape(out, [-1, 100*1])
    return out



def discriminator(inputs, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        output = tf.reshape(inputs, [-1,100,1,1])
        output = tf.layers.conv2d(output, filters=64, kernel_size=(3,1),
                                  strides=(2,1),
                                  padding='same',
                                  activation=tf.nn.leaky_relu)
        output = tf.layers.conv2d(output, filters=128, kernel_size=(3,1),
                                  strides=(2,1),
                                  padding='same',
                                  activation=tf.nn.leaky_relu)
        output = tf.layers.batch_normalization(output)

        output = tf.layers.conv2d(output, filters=256, kernel_size=(3,1),
                                  strides=(2,1),
                                  padding='same',
                                  activation=tf.nn.leaky_relu)
        output = tf.layers.batch_normalization(output)

        output = tf.layers.conv2d(output, filters=512, kernel_size=(3,1),
                                  strides=(2,1),
                                  padding='same',
                                  activation=tf.nn.leaky_relu)
        output = tf.layers.batch_normalization(output)

        output = tf.layers.dense(output,1)
    return output

batch_size = 64
z_dim = 128
learning_rate_ger = 5e-5
learning_rate_dis = 5e-5
lam = 10.
mode = 'gp'
clamp_lower = -0.01
clamp_upper = 0.01
# 定义loss

# 定义train operation
def build_graph():
    noise_dist = tf.contrib.distributions.Normal(0., 1.)
    z = noise_dist.sample((batch_size, z_dim))
    # gen = generator
    # dis = discriminator
    with tf.variable_scope('generator'):
        train = generator(z)

    real_data = tf.placeholder(dtype=tf.float32, shape=(None,100,1,1))
    true_logit = discriminator(real_data)
    fake_logit = discriminator(train, reuse=True)
    # data = np.concatenate((real_data, train))
    c_loss = tf.reduce_mean(fake_logit - true_logit)
    if mode is 'gp':
        alpha_dist = tf.contrib.distributions.Normal(0., 1.)
        alpha = alpha_dist.sample((batch_size, 1, 1, 1))
        interpolated = real_data + alpha*(train-real_data)
        inte_logit = discriminator(interpolated, reuse=True)
        gradients = tf.gradients(inte_logit, [interpolated,])[0]
        grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
        gradient_penalty = tf.reduce_mean((grad_l2-1)**2)
        gp_loss_sum = tf.summary.scalar("gp_loss", gradient_penalty)
        grad = tf.summary.scalar("grad_norm",tf.nn.l2_loss(gradients))
        c_loss += lam*gradient_penalty
    g_loss = tf.reduce_mean(-fake_logit)
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    c_loss_sum = tf.summary.scalar("c_loss", c_loss)
    img_sum = tf.summary.image("img", train, max_outputs=10)
    theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
    counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_g = tf.contrib.layers.optimize_loss(loss=g_loss, learning_rate= learning_rate_ger,
                                            optimizer=tf.train.RMSPropOptimizer,
                                            variables=theta_g, global_step=counter_g,
                                            summaries = ['gradient_norm'])
    counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_c = tf.contrib.layers.optimize_loss(loss=c_loss, learning_rate=learning_rate_dis,
                                            optimizer=tf.train.RMSPropOptimizer,
                                            variables=theta_c, global_step=counter_c,
                                            summaries = ['gradient_norm'])
    if mode is 'regular':
        clipped_var_c = [tf.assign(var, tf.clip_by_value(var, clamp_lower, clamp_upper)) for var in theta_c]
        with tf.control_dependencies([opt_c]):
            opt_c = tf.tuple(clipped_var_c)

    if not mode in ['gp','regular']:
        raise(NotImplementedError('Only two modes'))
    return opt_g, opt_c, real_data

log_dir = './log_wgan'
ckpt_dir = './ckpt_wgan'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
# max iter step, note the one step indicates that a Citers updates of critic and one update of generator
max_iter_step = 20000
Citers = 5
batch_size = 64
import random
def main():
    train_data, train_labels, test_data, test_labels = load_dataset()
    train_data = train_data.transpose((0, 2, 1))
    test_data = test_data.transpose((0, 2, 1))
    # print(train_data.shape, test_data.shape)
    train_data, test_data = scale_data(train_data, test_data)
    train_data = train_data.reshape(-1, 100, 1, 1)
    test_data = test_data.reshape(-1, 100, 1, 1)
    opt_g, opt_c, real_data = build_graph()
    merged_all = tf.summary.merge_all()
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

    def next_feed_dict():
        batch_size = 64
        temp = list(range(0, len(train_data)))
        random.shuffle(temp)
        idx = temp[0:batch_size]
        train_img = train_data[idx]
        feed_dict = {real_data: train_img}
        return feed_dict
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        for i in range(max_iter_step):
            if i < 25 or i % 500 == 0:
                citers = 100
            else:
                citers = Citers
            for j in range(citers):
                feed_dict = next_feed_dict()
                if i % 100 == 99 and j == 0:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, merged = sess.run([opt_c, merged_all], feed_dict=feed_dict,
                                         options=run_options, run_metadata=run_metadata)
                    summary_writer.add_summary(merged, i)
                    summary_writer.add_run_metadata(
                        run_metadata, 'critic_metadata {}'.format(i), i)
                else:
                    sess.run(opt_c, feed_dict=feed_dict)
            feed_dict = next_feed_dict()
            if i % 100 == 99:
                _, merged = sess.run([opt_g, merged_all], feed_dict=feed_dict,
                     options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged, i)
                summary_writer.add_run_metadata(
                    run_metadata, 'generator_metadata {}'.format(i), i)
            else:
                sess.run(opt_g, feed_dict=feed_dict)
            if i % 1000 == 999:
                saver.save(sess, os.path.join(
                    ckpt_dir, "model.ckpt"), global_step=i)

if __name__ == "__main__":
    main()
















