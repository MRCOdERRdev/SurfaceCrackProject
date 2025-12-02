import tensorflow as tf
import tf2onnx

h5_path = "MobileNetV2.h5"
onnx_path = "MobileNetV2.onnx"

model = tf.keras.models.load_model(h5_path)
print("Model loaded.")

spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="input"),)

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path=onnx_path
)

print("Conversion complete:", onnx_path)
