from typing import Dict

import tensorflow as tf
from absl import logging

from model import SequenceTagger
from train import LABELS, _create_model_input


def main():
    logging.info("Preprare model and preprocess fn")
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
            values=tf.constant(LABELS),
            value_dtype=tf.string,
            keys=tf.range(len(LABELS)),
        ),
        default_value="O",  # O
    )

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
    model.load_weights("./model/model")
    model(
        {
            "input_char": tf.keras.Input([None], dtype=tf.int64),
            "input_mask": tf.keras.Input([None], dtype=tf.int64),
            "input_script": tf.keras.Input([None], dtype=tf.int64),
        }
    )
    model.summary()

    model_input_fn = _create_model_input(vocab_table)

    @tf.function
    def inference(sentence: tf.Tensor):
        token = tf.strings.unicode_split(sentence, "UTF-8")
        model_input = model_input_fn(token)
        return model(model_input)[0]

    def print_results(sentence, inference_result):
        result = ""
        current_entity = ""

        for ch, label in zip(sentence, label_table.lookup(inference_result)[0].numpy().tolist()):
            label = label.decode("utf8")
            if label.startswith("B-"):
                if current_entity != "":
                    result += f":{current_entity}>"
                result += "<"
                current_entity = label[2:]

            if label == "O" and current_entity != "":
                result += f":{current_entity}>"
                current_entity = ""

            result += ch

        print(result)

    while True:
        sentence = input("문장 입력:")
        inference_result = inference(tf.constant([sentence]))
        print_results(sentence, inference_result)


if __name__ == "__main__":
    main()
