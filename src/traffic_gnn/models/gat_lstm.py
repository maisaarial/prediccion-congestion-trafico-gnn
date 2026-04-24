import tensorflow as tf
from tensorflow.keras import layers, Model


class SimpleGATLayer(layers.Layer):
    def __init__(self, units, activation="relu"):
        super(SimpleGATLayer, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True
        )

        self.a = self.add_weight(
            shape=(2 * self.units, 1),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, X, A):
        H = tf.matmul(X, self.W)

        N = tf.shape(H)[1]

        H_i = tf.tile(tf.expand_dims(H, axis=2), [1, 1, N, 1])
        H_j = tf.tile(tf.expand_dims(H, axis=1), [1, N, 1, 1])

        concat = tf.concat([H_i, H_j], axis=-1)

        e = tf.squeeze(tf.matmul(concat, self.a), axis=-1)
        e = tf.nn.leaky_relu(e)

        mask = tf.where(A > 0, 0.0, -1e9)
        attention = tf.nn.softmax(e + mask, axis=-1)

        H_out = tf.matmul(attention, H)

        return self.activation(H_out)


def build_gat_lstm_model(T, N, F, A, gat_units=32, lstm_units=64, learning_rate=0.001):
    inputs = layers.Input(shape=(T, N, F))

    gat = SimpleGATLayer(gat_units)

    outputs_gat = []
    for t in range(T):
        x_t = inputs[:, t, :, :]
        h_t = gat(x_t, A)
        outputs_gat.append(h_t)

    H = tf.stack(outputs_gat, axis=1)

    H = tf.transpose(H, perm=[0, 2, 1, 3])
    H = tf.reshape(H, (-1, T, gat_units))

    H = layers.LSTM(lstm_units)(H)

    outputs = layers.Dense(1)(H)
    outputs = tf.reshape(outputs, (-1, N, 1))

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )

    return model