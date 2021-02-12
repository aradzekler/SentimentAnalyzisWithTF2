import tensorflow as tf
from keras.engine.saving import load_model
import tensorflow as tf
from keras.engine.saving import load_model


print("prediction!")
classifier = load_model('lstm.h5')
classifier.compile(loss='binary_crossentropy',
               optimizer=tf.keras.optimizers.Adam(0.001),
               metrics=['accuracy'])
sample_text = "Awesome movie, the acting was incredible, I highly recommend!"
#predictions = sample_predict(sample_text, pad=True, model=classifier)
result = classifier.predict(["Awesome movie!"])
print("Probability this was a positive review: %.2f" % result)