import tensorflow as tf


class AgentVRP:
    VEHICLE_CAPACITY = 1.0

    def __init__(self, input):
        depot = input[0]
        loc = input[1]

        self.batch_size, self.n_loc, _ = loc.shape  # (batch_size, n_nodes, 2)

        # Coordinates of depot + other nodes
        self.coords = tf.concat((depot[:, None, :], loc), -2)  # 沿着一个维度连接张量
        self.demand = tf.cast(input[2], tf.float32)

        # Indices of graphs in batch  索引
        self.ids = tf.range(self.batch_size, dtype=tf.int64)[:, None]

        # State
        self.prev_a = tf.zeros((self.batch_size, 1), dtype=tf.float32)
        self.from_depot = self.prev_a == 0
        self.used_capacity = tf.zeros((self.batch_size, 1), dtype=tf.float32)

        # Nodes that have been visited will be marked with 1
        self.visited = tf.zeros((self.batch_size, 1, self.n_loc + 1), dtype=tf.uint8)

        # Step counter
        self.i = tf.zeros(1, dtype=tf.int64)

        # Constant tensors for scatter update (in step method)
        self.step_updates = tf.ones((self.batch_size, 1), dtype=tf.uint8)  # (batch_size, 1)
        self.scatter_zeros = tf.zeros((self.batch_size, 1), dtype=tf.int64)  # (batch_size, 1)

    @staticmethod
    def outer_pr(a, b):
        """Outer product of matrices  注：a shape[2,3], b shape[3,5] --> tf.einsum(a, b) shape[2, 5]
        """
        return tf.einsum('ki,kj->kij', a, b)

    def get_att_mask(self):
        """ Mask (batch_size, n_nodes, n_nodes) for attention encoder.
            We mask already visited nodes except depot
        """

        # We dont want to mask depot
        # [batch_size, 1, n_nodes] --> [batch_size, n_nodes-1]
        att_mask = tf.squeeze(tf.cast(self.visited, tf.float32), axis=-2)[:, 1:]  # 删除维数为1的维度

        # Number of nodes in new instance after masking
        cur_num_nodes = self.n_loc + 1 - tf.reshape(tf.reduce_sum(att_mask, -1), (-1, 1))  # [batch_size, 1]

        att_mask = tf.concat((tf.zeros(shape=(att_mask.shape[0], 1), dtype=tf.float32), att_mask), axis=-1)

        ones_mask = tf.ones_like(att_mask)

        # Create square attention mask from row-like mask
        att_mask = AgentVRP.outer_pr(att_mask, ones_mask) \
                   + AgentVRP.outer_pr(ones_mask, att_mask) \
                   - AgentVRP.outer_pr(att_mask, att_mask)

        return tf.cast(att_mask, dtype=tf.bool), cur_num_nodes

    def all_finished(self):
        """Checks if all games are finished
        """
        return tf.reduce_all(tf.cast(self.visited, tf.bool))  # 逻辑和

    def partial_finished(self):
        """Checks if partial solution for all graphs has been built, i.e. all agents came back to depot
            检查所有图的部分解是否已构建，即所有代理是否返回仓库
        """
        return tf.reduce_all(self.from_depot) and self.i != 0

    def get_mask(self):
        """ Returns a mask (batch_size, 1, n_nodes) with available actions.
            Impossible nodes are masked.
            返回一个具有可用操作的掩码(批大小，1,n个节点)。不可能的节点被屏蔽。
        """

        # Exclude depot
        visited_loc = self.visited[:, :, 1:]

        # Mark nodes which exceed vehicle capacity 标记超出车辆容量的节点
        exceeds_cap = self.demand + self.used_capacity > self.VEHICLE_CAPACITY

        # We mask nodes that are already visited or have too much demand 我们屏蔽已经访问过的节点或有太多需求的节点
        # Also for dynamical model we stop agent at depot when it arrives there (for partial solution)
        # 此外，对于动态模型，当代理到达仓库时，我们在那里停止代理(对于部分解决方案)
        mask_loc = tf.cast(visited_loc, tf.bool) | exceeds_cap[:, None, :] | (
                (self.i > 0) & self.from_depot[:, None, :])

        # We can choose depot if 1) we are not in depot OR 2) all nodes are visited
        # 如果1)我们不在仓库，或者2)所有节点都被访问，我们可以选择仓库
        mask_depot = self.from_depot & (tf.reduce_sum(tf.cast(mask_loc == False, tf.int32), axis=-1) > 0)
        return tf.concat([mask_depot[:, :, None], mask_loc], axis=-1)

    def step(self, action):
        # Update current state
        selected = action[:, None]

        self.prev_a = selected
        self.from_depot = self.prev_a == 0

        # We have to shift indices by 1 since demand doesn't include depot 由于需求不包括库存，我们不得不将指数移动1
        # 0-index in demand corresponds to the FIRST node 需求中的0-index对应于FIRST节点
        selected_demand = tf.gather_nd(self.demand,
                                       tf.concat([self.ids, tf.clip_by_value(self.prev_a - 1, 0, self.n_loc - 1)],
                                                 axis=1)
                                       )[:, None]  # (batch_size, 1)

        # We add current node capacity to used capacity and set it to zero if we return to the depot
        # 我们将当前节点容量添加到已使用容量中，并在返回仓库时将其设置为零
        self.used_capacity = (self.used_capacity + selected_demand) * (1.0 - tf.cast(self.from_depot, tf.float32))

        # Update visited nodes (set 1 to visited nodes) 更新已访问节点(设置1为已访问节点)
        # (batch_size, 1, 3)
        idx = tf.cast(tf.concat((self.ids, self.scatter_zeros, self.prev_a), axis=-1), tf.int32)[:, None, :]
        self.visited = tf.tensor_scatter_nd_update(self.visited, idx, self.step_updates)  # (batch_size, 1, n_nodes)

        self.i = self.i + 1

    @staticmethod
    def get_costs(dataset, pi):
        # Place nodes with coordinates in order of decoder tour  按解码器行程的顺序放置坐标节点
        loc_with_depot = tf.concat([dataset[0][:, None, :], dataset[1]], axis=1)  # (batch_size, n_nodes, 2)
        d = tf.gather(loc_with_depot, tf.cast(pi, tf.int32), batch_dims=1)

        # Calculation of total distance 总距离计算
        # Note: first element of pi is not depot, but the first selected node in the path
        # 注意:pi的第一个元素不是depot，而是路径中第一个选择的节点
        return (tf.reduce_sum(tf.norm(d[:, 1:] - d[:, :-1], ord=2, axis=2), axis=1)
                + tf.norm(d[:, 0] - dataset[0], ord=2, axis=1)  # Distance from depot to first selected node
                + tf.norm(d[:, -1] - dataset[0], ord=2,
                          axis=1))  # Distance from last selected node (!=0 for graph with longest path) to depot
