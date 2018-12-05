import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import static_bidirectional_rnn
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper
from tensorflow.python.ops.rnn_cell_impl import MultiRNNCell


class TextCNN:
    def __init__(self, sequence_length, num_classes,
                 text_vocab_size, text_embedding_size, pos_vocab_size, pos_embedding_size,
                 filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
        self.input_p1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_p1')
        self.input_p2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_p2')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        initializer = tf.keras.initializers.glorot_normal

        # Embedding layer
        with tf.device('/cpu:0'), tf.variable_scope("text-embedding"):
            self.W_text = tf.Variable(tf.random_uniform([text_vocab_size, text_embedding_size], -0.25, 0.25), name="W_text")
            self.text_embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)
            self.text_embedded_chars_expanded = tf.expand_dims(self.text_embedded_chars, -1)

        with tf.device('/cpu:0'), tf.variable_scope("position-embedding"):
            self.W_pos = tf.get_variable("W_pos", [pos_vocab_size, pos_embedding_size], initializer=initializer())
            self.p1_embedded_chars = tf.nn.embedding_lookup(self.W_pos, self.input_p1)
            self.p2_embedded_chars = tf.nn.embedding_lookup(self.W_pos, self.input_p2)
            self.p1_embedded_chars_expanded = tf.expand_dims(self.p1_embedded_chars, -1)
            self.p2_embedded_chars_expanded = tf.expand_dims(self.p2_embedded_chars, -1)

        self.embedded_chars_expanded = tf.concat([self.text_embedded_chars_expanded,
                                                  self.p1_embedded_chars_expanded,
                                                  self.p2_embedded_chars_expanded], 2)
        _embedding_size = text_embedding_size + 2*pos_embedding_size

        hidden_size = 128
        num_layer = 1
        with tf.variable_scope('input_encode'):
            def create_cell():
                if self.dropout_keep_prob < 1.0:
                    single_cell = lambda: BasicLSTMCell(hidden_size)
                    hidden = MultiRNNCell([single_cell() for _ in range(num_layer)])
                    hidden = DropoutWrapper(hidden, input_keep_prob=self.dropout_keep_prob,
                                            output_keep_prob=self.dropout_keep_prob)
                else:
                    single_cell = lambda: BasicLSTMCell(hidden_size)
                    hidden = MultiRNNCell([single_cell() for _ in range(num_layer)])
                return hidden

            self.init_hidden_fw = create_cell()
            self.init_hidden_bw = create_cell()

            outputs, hidden_fw, hidden_bw = static_bidirectional_rnn(self.init_hidden_fw, self.init_hidden_bw,
                                                                     self.embedded_chars_expanded,
                                                                     sequence_length=self.seq_length,
                                                                     dtype=tf.float32)  # outputs [(,256),..,(,256)]

            # get last layer state
            last_hidden_fw = hidden_fw[-1]  # (c, h) ((,128), (,128))
            last_hidden_bw = hidden_bw[-1]  # (c, h)
            self.last_hidden_state = tf.concat([tf.concat(last_hidden_fw, 1), tf.concat(last_hidden_bw, 1)], 1)  # (, 4*128)

            self.all_hidden_state = [tf.reshape(o, [-1, 1, self.init_hidden_fw.output_size + self.init_hidden_bw.output_size]) for o in outputs]  # [(,1,256),...(,1,256)]
            self.all_hidden_state = tf.concat(self.all_hidden_state, 1)  # (,30,256)

        with tf.variable_scope("decode_output"):
            batch_size = self.all_hidden_state.get_shape()[0]
            seq_length = self.all_hidden_state.get_shape()[1]
            att_size = self.all_hidden_state.get_shape()[2]

            source_hidden = tf.reshape(self.all_hidden_state, [-1, seq_length, 1, att_size])  # (B,30,1,256)
            attn_weight_list = []
            context_vec_list = []  # Results of attention reads will be stored here.

            for i in range(self.num_head):
                k = tf.get_variable("AttnK_%d" % i, [1, 1, att_size, att_size])
                v = tf.get_variable("AttnV_%d" % i, [att_size])
                conv_source_hidden = tf.nn.conv2d(source_hidden, k, [1, 1, 1, 1], "SAME")  # (B,30,1,256)

                with tf.variable_scope("Attention_%d" % i):
                    query = tf.layers.dense(self.last_hidden_state, att_size)  # (B, 256)
                    query = tf.reshape(query, [-1, 1, 1, att_size])  # (B,1,1,256)

                    # Attention mask is a softmax of v^T * tanh(...).
                    score = v * tf.tanh(conv_source_hidden + query)
                    s = tf.reduce_sum(score, [2, 3])  # (B, 30)

                    att_weight = tf.nn.softmax(s)
                    attn_weight_list.append(att_weight)

                    # Now calculate the attention-weighted context vector.
                    context_vec = tf.reduce_sum(tf.reshape(att_weight, [-1, seq_length, 1, 1]) * source_hidden, [1, 2])  # (B,256)
                    context_vec_list.append(tf.reshape(context_vec, [-1, att_size]))

            matrix = tf.get_variable("Out_Matrix", [att_size, self.num_class])  # (256,31)
            res = tf.matmul(context_vec_list[0], matrix)  # NOTE: here we temporarily assume num_head = 1

            bias_start = 0.0
            bias_term = tf.get_variable("Out_Bias", [self.num_class], initializer=tf.constant_initializer(bias_start))
            self.decode_output = [res + bias_term]  # (B,32)
            self.att_weight = attn_weight_list[0]








        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                conv = tf.layers.conv2d(self.embedded_chars_expanded, num_filters, [filter_size, _embedding_size],
                                        kernel_initializer=initializer(), activation=tf.nn.relu, name="conv")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(conv, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.variable_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.variable_scope("output"):
            self.logits = tf.layers.dense(self.h_drop, num_classes, kernel_initializer=initializer())
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
