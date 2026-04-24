import tensorflow as tf
from tensorflow.keras import layers, Model


class GCNLayer(layers.Layer):
    def __init__(self, units, activation="relu"):
        super(GCNLayer, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, X, A_hat):
        XW = tf.matmul(X, self.W)
        AXW = tf.matmul(A_hat, XW)
        return self.activation(AXW)


def build_gcn_gru_model(T, N, F, A_hat, gcn_units=32, gru_units=64, learning_rate=0.001):
    inputs = layers.Input(shape=(T, N, F))

    gcn = GCNLayer(gcn_units)

    outputs_gcn = []
    for t in range(T):
        x_t = inputs[:, t, :, :]
        h_t = gcn(x_t, A_hat)
        outputs_gcn.append(h_t)

    H = tf.stack(outputs_gcn, axis=1)

    H = tf.transpose(H, perm=[0, 2, 1, 3])
    H = tf.reshape(H, (-1, T, gcn_units))

    H = layers.GRU(gru_units)(H)

    outputs = layers.Dense(1)(H)
    outputs = tf.reshape(outputs, (-1, N, 1))

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )

    return model