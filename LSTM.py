import tensorflow as tf
from keras.layers import Input, LSTM, Dropout, Dense
from keras.models import Model


class Lstm(Model):
    def __init__(self, in_shape: tuple = (119,), out_shape: int = 2):
        super(Lstm, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

        model_input = Input(shape=self.in_shape)

        y = LSTM(units=1024, activation='tanh', return_sequences=True)(model_input)
        y = Dropout(0.2)(y)

        y = LSTM(units=1024, activation='tanh')(y)

        # Output layer for classification
        output = Dense(self.out_shape,
                       activation='softmax')(y)

        self.model = tf.keras.Model(inputs=model_input, outputs=output)

        self.initialize()
        self.model.summary()

    def call(self, inputs):
        return self.model(inputs)

    def initialize(self):
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer.kernel_initializer = tf.keras.initializers.GlorotNormal()
                layer.bias_initializer = tf.keras.initializers.Zeros()

    def build_graph(self):
        x = tf.keras.Input(shape=self.in_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


if __name__ == "__main__":
    cf = LSTM((118,), length_in=125973)
    cf.build_graph()
