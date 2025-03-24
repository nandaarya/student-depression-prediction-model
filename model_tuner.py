
import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from typing import NamedTuple, Dict, Text, Any
from keras_tuner.engine import base_tuner
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = "Depression"
NUMERIC_FEATURES = ["Academic Pressure", "Age", "CGPA", "Study Satisfaction", "Work/Study Hours"]
CATEGORICAL_FEATURES = ["Dietary Habits", "Family History of Mental Illness", 
                        "Financial Stress", "Gender", "Have you ever had suicidal thoughts ?", "Sleep Duration"]
NUM_EPOCHS = 10

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])

def transformed_name(key):
    return key.replace("/", "_").replace(" ", "_").lower() + "_xf"

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")

def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=LABEL_KEY)
    return dataset

def model_builder(hp, tf_transform_output):
    inputs = {}
    numeric_inputs = [tf.keras.Input(shape=(1,), name=transformed_name(f), dtype=tf.float32) for f in NUMERIC_FEATURES]
    for f in NUMERIC_FEATURES:
        inputs[transformed_name(f)] = numeric_inputs[NUMERIC_FEATURES.index(f)]
    
    concat_numeric = layers.concatenate(numeric_inputs)
    x = layers.Dense(hp.Int("dense_units_1", min_value=32, max_value=128, step=32), activation='relu')(concat_numeric)
    
    categorical_inputs = []
    for feature in CATEGORICAL_FEATURES:
        transformed_feature_name = transformed_name(feature)
        vocab_size = tf_transform_output.vocabulary_size_by_name(feature.replace(" ", "_").lower() + "_vocab") + 1
        
        categorical_input = tf.keras.Input(shape=(vocab_size,), name=transformed_feature_name, dtype=tf.float32)
        categorical_inputs.append(categorical_input)
        inputs[transformed_feature_name] = categorical_input
    
    concat_categorical = layers.concatenate(categorical_inputs)
    combined = layers.concatenate([x, concat_categorical])
    x = layers.Dense(hp.Int("dense_units_2", min_value=32, max_value=128, step=32), activation='relu')(combined)
    x = layers.Dropout(hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1))(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    return model

def tuner_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files, tf_transform_output, NUM_EPOCHS)
    eval_set = input_fn(fn_args.eval_files, tf_transform_output, NUM_EPOCHS)
    
    tuner = kt.BayesianOptimization(
        hypermodel=lambda hp: model_builder(hp, tf_transform_output),
        objective=kt.Objective('binary_accuracy', direction='max'),
        max_trials=30,
        directory=fn_args.working_dir,
        project_name="bayesian_tuning",
    )
    
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_set,
            "validation_data": eval_set,
            "epochs": NUM_EPOCHS,
        },
    )
