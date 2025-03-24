
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
import os
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = "Depression"
NUMERIC_FEATURES = ["Academic Pressure", "Age", "CGPA", "Study Satisfaction", "Work/Study Hours"]
CATEGORICAL_FEATURES = ["Dietary Habits", "Family History of Mental Illness", 
                        "Financial Stress", "Gender", "Have you ever had suicidal thoughts ?", "Sleep Duration"]

def transformed_name(key):
    """Convert feature name to match transformed feature in preprocessing."""
    return key.replace("/", "_").replace(" ", "_").lower() + "_xf"

def gzip_reader_fn(filenames):
    """Read compressed TFRecord dataset."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    """Load and parse transformed dataset."""
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=LABEL_KEY)
    return dataset

def model_builder(tf_transform_output):
    """Build the deep learning model using transformed features."""
    inputs = {}
    
    # Numeric Features (already normalized)
    numeric_inputs = [tf.keras.Input(shape=(1,), name=transformed_name(f), dtype=tf.float32) for f in NUMERIC_FEATURES]
    for f in NUMERIC_FEATURES:
        inputs[transformed_name(f)] = numeric_inputs[NUMERIC_FEATURES.index(f)]
    
    # Concatenate numeric inputs
    concat_numeric = layers.concatenate(numeric_inputs)
    x = layers.Dense(64, activation='relu')(concat_numeric)
    x = layers.Dense(32, activation='relu')(x)

    # Categorical Features (One-Hot Encoded)
    categorical_inputs = []
    for feature in CATEGORICAL_FEATURES:
        transformed_feature_name = transformed_name(feature)
        vocab_size = tf_transform_output.vocabulary_size_by_name(feature.replace(" ", "_").lower() + "_vocab") + 1
        
        categorical_input = tf.keras.Input(shape=(vocab_size,), name=transformed_feature_name, dtype=tf.float32)
        categorical_inputs.append(categorical_input)
        inputs[transformed_feature_name] = categorical_input

    # Concatenate categorical inputs
    concat_categorical = layers.concatenate(categorical_inputs)
    
    # Combine Numeric & Categorical Features
    combined = layers.concatenate([x, concat_categorical])
    x = layers.Dense(64, activation='relu')(combined)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.01),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    model.summary()
    return model

def _get_serve_tf_example_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        
        return model(transformed_features)
    
    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs):
    """Train and save the model."""
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')
    es = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max', patience=10)
    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(fn_args.serving_model_dir, "best_model.keras"),
        monitor='val_binary_accuracy',
        mode='max',
        save_best_only=True
    )
    
    # Load TFT transform graph
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # Load training and validation datasets
    train_set = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10)
    
    # Build and train the model
    model = model_builder(tf_transform_output)
    model.fit(
        x=train_set,
        validation_data=val_set,
        callbacks=[tensorboard_callback, es, mc],
        epochs=10)

    signatures = {
    "serving_default": _get_serve_tf_example_fn(
        model, tf_transform_output
    ).get_concrete_function(
        tf.TensorSpec(
            shape=[None], 
            dtype=tf.string, 
            name="examples"
        )
    )}
    
    # Save final model
    tf.saved_model.save(model, fn_args.serving_model_dir, signatures=signatures)
