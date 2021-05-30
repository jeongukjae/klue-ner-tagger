from typing import Tuple

import tensorflow as tf
import tensorflow_addons as tfa


class SequenceTagger(tf.keras.Model):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        embedding_size: int,
        max_uscript_size: int,
        uscript_embedding_size: int,
        num_filters: int,
        max_ngram: int,
        **kwargs,
    ):
        # max_uscript_size: https://unicode-org.github.io/icu-docs/apidoc/released/icu4c/uscript_8h.html
        super().__init__(**kwargs)

        self.embedding_table = tf.keras.layers.Embedding(vocab_size, embedding_size, name="embedding_table")
        self.uscript_table = tf.keras.layers.Embedding(max_uscript_size, uscript_embedding_size, name="uscript_table")
        self.conv_layers = [
            tf.keras.layers.Conv1D(
                num_filters,
                ngram,
                padding="same",
                activation="relu",
                name=f"conv_{ngram}",
            )
            for ngram in range(1, max_ngram + 1)
        ]

        self.decoder = tfa.layers.CRF(num_classes, name="decoder")

    def call(self, inputs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        embedding = self.embedding_table(inputs["input_char"])
        uscript_embedding = self.uscript_table(inputs["input_script"])

        context_representation = tf.concat(
            [conv(embedding) for conv in self.conv_layers] + [uscript_embedding],
            axis=-1,
        )

        output = self.decoder(context_representation, mask=inputs["input_mask"])
        return output

    def train_step(self, inputs):
        x, y = inputs

        with tf.GradientTape() as tape:
            decoded, potentials, sequence_length, chain_kernel = self(x, training=True)
            loss = -tfa.text.crf.crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        y = tf.map_fn(
            lambda x: x[0][: x[1]],
            [y, sequence_length],
            fn_output_signature=tf.RaggedTensorSpec([None], dtype=tf.int32),
        )
        decoded = tf.map_fn(
            lambda x: x[0][: x[1]],
            [decoded, sequence_length],
            fn_output_signature=tf.RaggedTensorSpec([None], dtype=tf.int32),
        )
        self.compiled_metrics.update_state(y, decoded)
        return {"loss": loss, **{m.name: m.result() for m in self.metrics}}

    def test_step(self, data):
        x, y = data
        decoded, potentials, sequence_length, chain_kernel = self(x, training=False)
        loss = -tfa.text.crf.crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]

        y = tf.map_fn(
            lambda x: x[0][: x[1]],
            [y, sequence_length],
            fn_output_signature=tf.RaggedTensorSpec([None], dtype=tf.int32),
        )
        decoded = tf.map_fn(
            lambda x: x[0][: x[1]],
            [decoded, sequence_length],
            fn_output_signature=tf.RaggedTensorSpec([None], dtype=tf.int32),
        )
        self.compiled_metrics.update_state(y, decoded)
        return {"loss": loss, **{m.name: m.result() for m in self.metrics}}
