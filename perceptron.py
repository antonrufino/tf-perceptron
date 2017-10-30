import tensorflow as tf


class Perceptron:
    def __init__(self, num_features):
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.int32)
        w = tf.Variable(tf.random_uniform([1, num_features]))
        b = tf.Variable(tf.random_uniform([1]))

        y_ = tf.matmul(w, tf.cast(self.x, tf.float32)) + b
        self.y_ = tf.cond(
            y_[0, 0] > 0.,
            lambda: tf.constant(1, dtype=tf.int32),
            lambda: tf.constant(0, dtype=tf.int32)
        )

        self.adjust_w = w.assign_add(
            tf.cast(self.y - self.y_, tf.float32) * tf.transpose(self.x))
        self.adjust_b = b.assign_add([tf.cast(self.y - self.y_, tf.float32)])


if __name__ == '__main__':
    x = [
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]]
    ]
    y = [0, 0, 1, 1]

    x = tf.constant(x)
    y = tf.constant(y)

    dataset = tf.contrib.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(4)
    dataset = dataset.repeat(20)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    model = Perceptron(2)

    sess = tf.Session()

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    while True:
        try:
            input_x, input_y = sess.run(next_element)
            y_, _, _ = sess.run([model.y_, model.adjust_w, model.adjust_b],
                                feed_dict={model.x: input_x, model.y: input_y})
            print(input_x, y_, input_y)
        except tf.errors.OutOfRangeError:
            break
