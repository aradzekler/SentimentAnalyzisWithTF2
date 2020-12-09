import os
import tempfile
import tensorflow_datasets as tfds
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow_core.python import InteractiveSession, ConfigProto

# solved some issues with GPU
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)

train_ds, test_ds = dataset['train'], dataset['test']

encoder = info.features['text'].encoder

BUFFER_SIZE = 10000
BATCH_SIZE = 16

padded_shapes = ([None], ())

# Shuffeling our dataset and giving it padded batching - This transformation combines multiple
# consecutive elements of the input dataset into a single element. as combines words into a sentence.
train_ds = train_ds.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE,
                                                      padded_shapes=padded_shapes)
test_ds = test_ds.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE,
                                                    padded_shapes=padded_shapes)

# Callbacks
# a path for the model.
model_dir = tempfile.gettempdir()
model_ver = 1.0
export_path = os.path.join(model_dir, str(model_ver))

# will create a file checkpoint for our model, it will overwrite it every run until we will find the best model
checkpoint = ModelCheckpoint(filepath=export_path + '\model',
                             monitor='val_loss',  # monitor our progress by loss value.
                             mode='min',  # smaller loss is better, we try to minimize it.
                             save_best_only=True,
                             verbose=1)

# if our model accuracy (loss) is not improving over 3 epochs, stop the training, something is fishy
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

# if our loss is not improving, try to reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [checkpoint, earlystop, reduce_lr]

# Our Keras Sequential() model.
_model = tf.keras.Sequential([tf.keras.layers.Embedding(encoder.vocab_size, 64),
                              tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
                              tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
                              tf.keras.layers.Dense(64, activation='elu'),
                              tf.keras.layers.Dense(1, activation='sigmoid')])
_model.summary()
_model.compile(loss='binary_crossentropy',
               optimizer=tf.keras.optimizers.Adam(0.001),
               metrics=['accuracy'])
history = _model.fit(train_ds, epochs=5, validation_data=test_ds, validation_steps=30,
                     callbacks=callbacks)


# adds a zero padding depends on the size wanted
def pad_to_size(vec, size):
	zeroes = [0] * (size - len(vec))
	vec.extend(zeroes)
	return vec


def sample_predict(sentence, pad, model):
	encoded_sample_pred_text = encoder.encode(sentence)
	if pad:
		encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
	encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
	_predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))
	return _predictions


sample_text = "Awesome movie, the acting was incredible, I highly recommend!"
predictions = sample_predict(sample_text, pad=True, model=_model) * 100

print("Probability this was a positive review: %.2f" % predictions)
