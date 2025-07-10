# tracknet_converter.py
import os
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def create_tracknet_model(height=360, width=640):
    """Create TrackNet model architecture"""
    print("Creating TrackNet model...")
    # Input is 3 consecutive frames (RGB)
    input_layer = Input(shape=(height, width, 9))

    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)

    # Decoder
    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

    # Output - heatmap
    output_layer = Conv2D(1, (1, 1), activation='sigmoid')(conv7)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Model created.")
    return model

def convert_h5_model(input_path, output_path, try_load_weights=True):
    """
    Convert old TrackNet model to TensorFlow 2 compatible model
    
    Args:
        input_path: Path to old model file
        output_path: Path to save converted model
        try_load_weights: Whether to try loading weights from old model
    """
    print(f"Converting model from {input_path} to {output_path}...")
    
    # Create new model
    model = create_tracknet_model()
    
    if try_load_weights and os.path.exists(input_path):
        try:
            # Try loading the full model
            print("Trying to load full model...")
            loaded_model = load_model(input_path, compile=False)
            print("Successfully loaded full model.")
            model = loaded_model
        except Exception as e:
            print(f"Failed to load full model: {e}")
            
            try:
                # Try loading just the weights
                print("Trying to load just weights...")
                model.load_weights(input_path)
                print("Successfully loaded weights.")
            except Exception as e:
                print(f"Failed to load weights: {e}")
                print("WARNING: Using model with random initialization!")
    
    # Save model in TF2 format
    model.save(output_path, save_format='h5')
    print(f"Model saved to {output_path}")
    
    return model

def test_model(model_path):
    """Test if model can be loaded and used for prediction"""
    try:
        model = load_model(model_path, compile=False)
        print("Model load test successful!")
        
        # Test with random input
        import numpy as np
        test_input = np.random.random((1, 360, 640, 9))
        pred = model.predict(test_input)
        print(f"Prediction shape: {pred.shape}")
        return True
    except Exception as e:
        print(f"Model test failed: {e}")
        return False

if __name__ == "__main__":
    # Path to original model
    original_model = "/home/jcperez/Sergio/TFG/src/weights/model_tennis.h5"
    # Path to save converted model
    converted_model = "tracknet_converted.h5"
    
    # Convert model
    model = convert_h5_model(original_model, converted_model)
    
    # Test converted model
    test_model(converted_model)