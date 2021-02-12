import tensorflow as tf
import tensorflow_datasets as tfds


def sample_predict(sentence, pad, model):
	encoded_sample_pred_text = encoder.encode(sentence)
	if pad:
		encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
	encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
	_predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))
	return _predictions


# adds a zero padding depends on the size wanted
def pad_to_size(vec, size):
	zeroes = [0] * (size - len(vec))
	vec.extend(zeroes)
	return vec


dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
encoder = info.features['text'].encoder


classifier = tf.keras.models.load_model('lstm.h5')

percent_symbol = "%"


sample_text = "what a great movie! I really enjoyed it"
predictions = sample_predict(sample_text, pad=True, model=classifier) * 100
print("Probability this was a positive review: %.1f" % predictions + percent_symbol)
