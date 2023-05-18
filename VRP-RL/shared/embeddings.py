import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


# class Embedding(object):
class Embedding:
    """
    This class is the base class for embedding the input graph.
    """

    def __init__(self, emb_type, embedding_dim):
        self.emb_type = emb_type
        self.embedding_dim = embedding_dim

    def __call__(self, input_pnt):
        # returns the embeded tensor. Should be implemented in child classes
        pass


class LinearEmbedding(Embedding):
    """
    This class implements linear embedding. It is only a mapping
    to a higher dimensional space.
    """

    def __init__(self, embedding_dim, _scope=''):
        """
        Input:
            embedding_dim: embedding dimension
        """
        # super(XXX, self).init(): 对继承自父类的属性进行初始化，并且用父类的初始化方法初始化继承的属性。
        super(LinearEmbedding, self).__init__('linear', embedding_dim)  # 使用父类的初始化方法来初始化子类的embedding_dim属性
        # self.project_emb = tf.layers.Conv1D(embedding_dim, 1, _scope=_scope + 'Embedding/conv1d')
        self.project_emb = tf.keras.layers.Conv1D(embedding_dim, 1)

    def __call__(self, input_pnt):
        # emb_inp_pnt: [batch_size, max_time, embedding_dim]
        emb_inp_pnt = self.project_emb(input_pnt)
        # emb_inp_pnt = tf.Print(emb_inp_pnt,[emb_inp_pnt])
        return emb_inp_pnt


if __name__ == "__main__":
    """
    tensorflow 1.x由于是基于静态图机制（Graph Execution），需要先构造图，然后才真正运行，因此需要用显示调用Session后，才会真正触发计算。对调试代码非常不利。
    tensorflow 2.x默认是基于动态图机制（Eager Execution），就像常规函数一样，调用时就触发计算。对调试代码非常方便。
    """
    # tf1中session部分代码，可以全部去掉
    # sess = tf.InteractiveSession()
    # input_pnt = tf.random_uniform([2, 10, 2])
    input_pnt = tf.random.uniform([2, 10, 2])
    Embedding = LinearEmbedding(128)
    emb_inp_pnt = Embedding(input_pnt)
    # sess.run(tf.global_variables_initializer())
    # print(sess.run([emb_inp_pnt, tf.shape(emb_inp_pnt)]))
    print(emb_inp_pnt)

    # le = LinearEmbedding(128)
    # print('le', dir(le))
