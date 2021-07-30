import tensorflow as tf
import numpy as np

model = tf.saved_model.load('models/1627473931')
print('loaded')
infer = model.signatures["serving_default"]
print(infer.structured_input_signature)
print(infer.structured_outputs)
r = infer(**{
    "Input-Segment": tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='float'),
    "Input-Token": tf.constant([[ 101, 7942, 2434, 1353, 7668,  749,  671,  702, 7309, 7579,  102 ]], dtype='float'),
    })
# r = model.signatures["serving_default"](**{
#     "input": tf.constant([[1,2,300000]], dtype='float')
#     })
print(r)