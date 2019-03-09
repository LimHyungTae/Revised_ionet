import numpy as np
import tensorflow as tf
import pprint

class IONet:
    def __init__(self, args): # batch_size, input_size,sequence_length, hidden_size, output_size):
        self.batch_size = args.batch_size
        self.input_size = args.num_uwb
        self.preprocessing_size = args.preprocessing_output_size
        self.first_layer_output_size = args.first_layer_output_size
        self.second_layer_output_size = args.second_layer_output_size
        self.output_size = args.output_size
        self.sequence_length = args.sequence_length
        self.output_type = args.output_type
        self.is_multimodal = args.is_multimodal
        self.network_type = args.network_type

        self.fc_layer_output_size = args.fc_layer_size
        self.dropout_prob = args.dropout_prob
        self.clip = args.clip

        self.set_placeholders_for_non_multimodal()
        self.set_stacked_bi_LSTM()

    def set_placeholders_for_non_multimodal(self):
        self.X_data = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.sequence_length, self.input_size],
                                     name='input_placeholder')

##################################################
            #Builing RiTA's paper
##################################################

    def set_stacked_bi_LSTM(self):
        with tf.variable_scope("Stacked_bi_lstm1"):
            # outputs : tuple
            first_layer_output_num = 100
            cell_forward1 = tf.contrib.rnn.BasicLSTMCell(num_units=first_layer_output_num)
            cell_backward1 = tf.contrib.rnn.BasicLSTMCell(num_units=first_layer_output_num)

            # outputs : tuple
            outputs, _states = tf.nn.bidirectional_dynamic_rnn(cell_forward1, cell_backward1, self.X_data,
                                                               dtype=tf.float32)
            # outputs = tf.concat([outputs[0], outputs[1]], axis=1)
            outputs = tf.concat([outputs[0], outputs[1]], axis=2)

        with tf.variable_scope("Stacked_bi_lstm2"):
            cell_forward2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.output_size)
            cell_backward2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.output_size)

            # outputs : tuple
            outputs, _states = tf.nn.bidirectional_dynamic_rnn(cell_forward2, cell_backward2, outputs, dtype=tf.float32)
            outputs = tf.concat([outputs[0], outputs[1]], axis=2)

        self.output = tf.reshape(outputs, [-1, self.sequence_length*2*self.output_size])


##################################################
            #Builing RO Nets
##################################################

    def set_loss_terms(self):
        print ("Bulding loss terms...")
        self.error_btw_gt_and_pred = tf.reduce_mean(tf.square(self.position_gt - self.pose_pred))
        print ("Complete!")
    def build_loss(self, lr, lr_decay_rate, lr_decay_step):
        self.init_lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        batch_size = self.batch_size

        with tf.variable_scope('lstm_loss'):
            self.loss = self.error_btw_gt_and_pred
            tf.summary.scalar('lstm_loss', self.loss)

        with tf.variable_scope('train'):
            self.global_step = tf.contrib.framework.get_or_create_global_step()

            self.cur_lr = tf.train.exponential_decay(self.init_lr,
                                                     global_step=self.global_step,
                                                     decay_rate=self.lr_decay_rate,
                                                     decay_steps=self.lr_decay_step)

            tf.summary.scalar('global learning rate', self.cur_lr)

            self.optimizer = tf.train.AdamOptimizer(learning_rate= self.cur_lr)
            '''Gradient clipping parts'''
            # gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
            # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            # self.optimize = self.optimizer.apply_gradients(zip(gradients, variables))
            self.optimize = self.optimizer.minimize(self.loss, global_step=self.global_step)

