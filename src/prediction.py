def predict_delivery_minutes(pipeline, input_data):
    return float(pipeline.predict(input_data)[0])