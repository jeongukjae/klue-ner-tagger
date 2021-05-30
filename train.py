from typing import Dict

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import tfds_korean.klue_ner
from absl import logging

from model import SequenceTagger

LABELS = [label for entity in ["PS", "LC", "OG", "DT", "TI", "QT"] for label in [f"B-{entity}", f"I-{entity}"]] + ["O"]


def main():
    logging.info("Load dataset")
    ds = tfds.load("klue_ner")
    train_ds, dev_ds = ds["train"], ds["dev"]
    print("train example:", len(train_ds), ", dev example:", len(dev_ds))

    logging.info("Create vocab")
    _create_vocab(
        train_ds.map(lambda x: tf.strings.reduce_join(x["tokens"], separator="")).batch(64),
        "vocab.txt",
    )

    logging.info("Preprare datasets")
    vocab_table = tf.lookup.StaticVocabularyTable(
        initializer=tf.lookup.TextFileInitializer(
            filename="vocab.txt",
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
        ),
        num_oov_buckets=1,
    )
    label_table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(LABELS),
            key_dtype=tf.string,
            values=tf.range(len(LABELS)),
        ),
        default_value=len(LABELS) - 1,  # O
    )

    train_ds = (
        train_ds.shuffle(30000, reshuffle_each_iteration=True)
        .apply(tf.data.experimental.dense_to_ragged_batch(batch_size=32, drop_remainder=True))
        .map(_map_model_input(vocab_table, label_table))
    )
    dev_ds = dev_ds.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=100)).map(
        _map_model_input(vocab_table, label_table)
    )

    print(train_ds.element_spec)

    logging.info("Initialize model")
    model = SequenceTagger(
        num_classes=int(label_table.size()),
        vocab_size=vocab_table.size(),
        embedding_size=64,
        max_uscript_size=106,
        uscript_embedding_size=8,
        num_filters=16,
        max_ngram=10,
        name="ner_tager",
    )
    model(
        {
            "input_char": tf.keras.Input([None], dtype=tf.int64),
            "input_mask": tf.keras.Input([None], dtype=tf.int64),
            "input_script": tf.keras.Input([None], dtype=tf.int64),
        }
    )
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.PolynomialDecay(
                0.03,
                decay_steps=1000,
                end_learning_rate=0.005,
                power=1,
            )
        ),
        metrics=[
            tf.keras.metrics.Accuracy(),
            F1Score(num_classes=len(LABELS), average="macro"),
        ],
    )

    logging.info("Train model")
    model.fit(train_ds, validation_data=dev_ds, epochs=3, callbacks=[tf.keras.callbacks.TensorBoard()])
    model.save_weights("./model/model")


def _create_vocab(ds: tf.data.Dataset, output_filename="vocab.txt"):
    unique_chars = set()
    for string_tensor in ds:
        string = tf.strings.reduce_join(string_tensor, separator="").numpy().decode("utf8")
        unique_chars.update(string)
    unique_chars = ["<pad>", "<s>", "</s>"] + list(unique_chars)
    print("Total unique chars:", len(unique_chars))

    with open(output_filename, "w") as f:
        for char in unique_chars:
            print(char, file=f)


def _map_model_input(vocab_table: tf.lookup.StaticVocabularyTable, label_table: tf.lookup.StaticHashTable):
    model_input_fn = _create_model_input(vocab_table)

    def _inner(x: Dict[str, tf.Tensor]):
        return model_input_fn(x["tokens"]), label_table.lookup(x["labels"]).to_tensor()

    return _inner


def _create_model_input(vocab_table: tf.lookup.StaticVocabularyTable):
    def _inner(token: tf.Tensor):
        input_char = vocab_table.lookup(token).to_tensor()
        input_script = tf.strings.unicode_script(tf.strings.unicode_decode(token, "UTF-8")).merge_dims(1, 2).to_tensor()
        input_mask = tf.ragged.map_flat_values(tf.ones_like, token, dtype=tf.bool).to_tensor()

        return {
            "input_char": input_char,
            "input_mask": input_mask,
            "input_script": input_script,
        }

    return _inner


class F1Score(tfa.metrics.F1Score):
    def update_state(self, y_true: tf.RaggedTensor, y_pred: tf.RaggedTensor, *args, **kwargs):
        y_true = tf.one_hot(y_true.values, depth=self.num_classes)
        y_pred = tf.one_hot(y_pred.values, depth=self.num_classes, dtype=tf.float32)

        return super().update_state(y_true, y_pred, *args, **kwargs)


if __name__ == "__main__":
    main()
