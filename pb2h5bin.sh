tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --signature_name=serving_default \
    --saved_model_tags=serve \
    /mobilenet/saved_model \
    /mobilenet/web_model

    tensorflowjs_converter \            
    --input_format=tf_saved_model \
    --output_node_names='webModels' \
    --saved_model_tags=serve \
    ./models/1627553771 \
    ./web_model/web_model