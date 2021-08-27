
# Load saved model
import kashgari
from kashgari.utils import convert_to_saved_model

loaded_model = kashgari.utils.load_model('per_ner.h5')

convert_to_saved_model(loaded_model, 'models/')