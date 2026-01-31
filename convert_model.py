import tensorflow as tf
import numpy as np
import os

model_path = "plant_disease_finetuned.keras"
tflite_path = "plant_disease.tflite"

if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    try:
        # 1. Load the model
        model = tf.keras.models.load_model(model_path)

        # 2. Determine Input Shape automatically
        # (This handles the (128,128,3) logic dynamically)
        input_shape = model.inputs[0].shape
        dtype = model.inputs[0].dtype
        
        # Ensure batch dimension is 1 if it's None (standard for Keras)
        if input_shape[0] is None:
            input_shape = (1, *input_shape[1:])
            
        print(f"Detected input shape: {input_shape}")

        # 3. Create a "Concrete Function"
        # This records the graph execution, bypassing Keras serialization bugs
        @tf.function
        def serve(x):
            return model(x)

        concrete_func = serve.get_concrete_function(tf.TensorSpec(input_shape, dtype))

        # 4. Convert from the Concrete Function
        print("Converting to TFLite via Concrete Function trace...")
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        
        # Enable optimizations to reduce file size (Essential for Free Tier)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()

        # 5. Save the file
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        
        print(f"✅ Success! Created '{tflite_path}'")

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        # Print full error for debugging if this fails
        import traceback
        traceback.print_exc()

else:
    print(f"❌ Error: Could not find {model_path}")