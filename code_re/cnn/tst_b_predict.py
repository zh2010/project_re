# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import subprocess

from code_re.cnn import data_helpers
from code_re.cnn import utils
from code_re.cnn.configure import FLAGS
from code_re.config import Data_PATH, test_b_data_path


def evaluation():

    checkpoint_dir = 'runs/1543944404/checkpoints'
    
    # Map data into vocabulary
    text_path = os.path.join(checkpoint_dir, "..", "text_vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)

    # Map data into position
    position_path = os.path.join(checkpoint_dir, "..", "pos_vocab")
    position_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(position_path)

    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            input_p1 = graph.get_operation_by_name("input_p1").outputs[0]
            input_p2 = graph.get_operation_by_name("input_p2").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            for file_name in os.listdir(os.path.join(Data_PATH, 'test_b_processed_data')):
                pred_list = []
                with open(os.path.join(Data_PATH, 'test_b_processed_data', file_name)) as f:
                    for line in f:
                        sent_cut, e1, e2, pos1, pos2, e1_id, e2_id = line.strip().split('\t')

                        x_text = [sent_cut]
                        x = np.array(list(text_vocab_processor.transform(x_text)))

                        pos1 = [pos1]
                        pos2 = [pos2]
                        p1 = np.array(list(position_vocab_processor.transform(pos1)))
                        p2 = np.array(list(position_vocab_processor.transform(pos2)))

                        preds = sess.run(predictions, {input_text: x,
                                                       input_p1: p1,
                                                       input_p2: p2,
                                                       dropout_keep_prob: 1.0})

                        pred_label = utils.label2class[preds[0]]

                        pred_list.append([pred_label, e1_id, e2_id])

                existed_pair = set()
                if not os.path.exists(os.path.join(Data_PATH, 'submit')):
                    os.makedirs(os.path.join(Data_PATH, 'submit'))

                with open(os.path.join(Data_PATH, 'submit', file_name.replace('sample', 'ann')), 'w') as fout:
                    with open(os.path.join(test_b_data_path, file_name.replace('sample', 'ann'))) as fann:
                        for line in fann:
                            fout.write(line)
                    for idx, (pred_label, e1_id, e2_id) in enumerate(pred_list):
                        if pred_label == 'other':
                            continue
                        fout.write('R{}\t{} Arg1:{} Arg2:{}\n'.format(str(idx), pred_label, e1_id, e2_id))
                        existed_pair.add('{} {} {}'.format(pred_label, e1_id, e2_id))


def main(_):
    evaluation()


if __name__ == "__main__":
    tf.app.run()