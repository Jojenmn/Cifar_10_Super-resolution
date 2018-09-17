"""Tests for cifar10 input_image."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt

import os

import tensorflow as tf

import cifar10_input_image


class test_input_image_test(tf.test.TestCase):

    def test(self):
        data_dir = '/home/jojen/tensorflow-python3.5/cifar-10/data'
        # images, labels = cifar10_input_image.inputs(False, data_dir, 1)
        filenames = [_ for _ in os.listdir(data_dir) if _.endswith('.jpg')]
        if len(filenames) == 0:
            raise Exception('No files in the input directory.')
        filenames = [os.path.join(data_dir, _) for _ in filenames]
        filename_queue = tf.train.string_input_producer(filenames)

        reader = tf.WholeFileReader(name='image_reader')
        key, value = reader.read(filename_queue)
        data_dir_len = len(data_dir)
        label = tf.substr(key, data_dir_len + 1, 1)
        label = tf.string_to_number(label)
        label = tf.expand_dims(label, 0)
        # imagename = tf.cast(imagename, tf.int32)
        # imagename.set_shape([1])
        # imagename = tf.string_to_number(key)
        # imagename = tf.as_string(imagename)
        # imagename = tf.compat.as_str_any(key.eval())
        # imagename = key.eval
        # _, imagename = os.path.split(imagename)
        # label = ord(imagename[0]) - 48

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # sess.run(tf.Print(imagename, [imagename]))
            # print(key.eval())
            # for x in key.eval():
            #     print(x)
            print(label.get_shape())
            print(label.eval())
            print(data_dir_len)

            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
  tf.test.main()
