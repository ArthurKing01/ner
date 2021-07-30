import tensorflowjs as tfjs
from tensorflow.keras import models
newModel = models.load_model('/Users/didi/Documents/project3/Chinese_Time_Recogniztion/time_ner.h5/model_weights.h5')
# loaded_model = kashgari.utils.load_model('/Users/didi/Documents/project3/Chinese_Time_Recogniztion/time_ner.h5')

# tfjs.converters.save_keras_model(newModel, '/Users/didi/Documents/project3/Chinese_Time_Recogniztion/outJs')