
# Load saved model
import kashgari
from kashgari.utils import convert_to_saved_model
import argparse
import time

parser = argparse.ArgumentParser(description="your script description") 
parser.add_argument('--model_path', '-m', required=False, type=str)

args = parser.parse_args()

model_path = args.model_path if args.model_path else 'per_ner.h5'

name = model_path.split('.h5')[0]


loaded_model = kashgari.utils.load_model(model_path)

convert_to_saved_model(loaded_model, 'models/', name + '_' + str(int(time.time())))