import tensorflow as tf
from tms.vrp_am.layers import MultiHeadAttention


class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    """Feed-Forward Sublayer: fully-connected Feed-Forward network,
    built based on MHA vectors from MultiHeadAttention layer with skip-connections

        Args:
            num_heads: number of attention heads in MHA layers.
            input_dim: embedding size that will be used as d_model in MHA layers.
            feed_forward_hidden: number of neuron units in each FF layer.

        Call arguments:
            x: batch of shape (batch_size, n_nodes, node_embedding_size).
            mask: mask for MHA layer

        Returns:
               outputs of shape (batch_size, n_nodes, input_dim)

    """

    def __init__(self, input_dim, num_heads, feed_forward_hidden=512, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(n_heads=num_heads, d_model=input_dim, name='MHA')
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1', trainable=True)
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2', trainable=True)
        self.ff1 = tf.keras.layers.Dense(feed_forward_hidden, name='ff1')
        self.ff2 = tf.keras.layers.Dense(input_dim, name='ff2')

    def call(self, x, mask=None):
        mha_out = self.mha(x, x, x, mask)
        sc1_out = tf.keras.layers.Add()([x, mha_out])
        bn1_out = self.bn1(sc1_out, training=True)

        ff1_out = self.ff1(bn1_out)
        relu1_out = tf.keras.activations.relu(ff1_out)
        ff2_out = self.ff2(relu1_out)
        sc2_out = tf.keras.layers.Add()([bn1_out, ff2_out])
        bn2_out = self.bn2(sc2_out, training=True)

        return bn2_out


class GraphAttentionEncoder(tf.keras.layers.Layer):
    """Graph Encoder, which uses MultiHeadAttentionLayer sublayer.

        Args:
            input_dim: embedding size that will be used as d_model in MHA layers.
            num_heads: number of attention heads in MHA layers.
            num_layers: number of attention layers that will be used in encoder.
            feed_forward_hidden: number of neuron units in each FF layer.

        Call arguments:
            x: tuples of 3 tensors:  (batch_size, 2), (batch_size, n_nodes-1, 2), (batch_size, n_nodes-1)
            First tensor contains coordinates for depot, second one is for coordinates of other nodes,
            Last tensor is for normalized demands for nodes except depot

            mask: mask for MHA layer

        Returns:
               Embedding for all nodes + mean embedding for graph.
               Tuples ((batch_size, n_nodes, input_dim), (batch_size, input_dim))
    """

    def __init__(self, input_dim, num_heads, num_layers, feed_forward_hidden=512):
        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feed_forward_hidden = feed_forward_hidden

        # initial embeddings (batch_size, n_nodes-1, 2) --> (batch-size, input_dim), separate for depot and other nodes
        self.init_embed_depot = tf.keras.layers.Dense(self.input_dim, name='init_embed_depot')  # nn.Linear(2, embedding_dim)
        self.init_embed = tf.keras.layers.Dense(self.input_dim, name='init_embed')

        self.mha_layers = [MultiHeadAttentionLayer(self.input_dim, self.num_heads, self.feed_forward_hidden)
                           for _ in range(self.num_layers)]

    def call(self, x, mask=None):
        assert mask is None, "TODO mask not yet supported!"

        x = tf.concat((self.init_embed_depot(x[0])[:, None, :],  # (batch_size, 2) --> (batch_size, 1, 2)
                       self.init_embed(tf.concat((x[1], x[2][:, :, None]), axis=-1))
                       # (batch_size, n_nodes-1, 2) + (batch_size, n_nodes-1)
                       ), axis=1)  # (batch_size, n_nodes, input_dim)

        # stack attention layers
        for i in range(self.num_layers):
            x = self.mha_layers[i](x)

        output = (x, tf.reduce_mean(x, axis=1))
        return output  # (embeds of nodes, avg graph embed)=((batch_size, n_nodes, input), (batch_size, input_dim))
