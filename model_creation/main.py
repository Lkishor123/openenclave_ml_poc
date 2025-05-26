import onnx
from onnx import helper
from onnx import TensorProto

# Define input and output
X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, 2]) # Batch size, 2 features
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, 2])

# Create an Identity node
node_def = helper.make_node(
    'Identity', # node operator_type
    ['input'],  # inputs
    ['output'], # outputs
)

# Create the graph (model)
graph_def = helper.make_graph(
    [node_def],
    'simple-identity-model',
    [X],
    [Y],
)

# Create the model
model_def = helper.make_model(graph_def, producer_name='onnx-example')
model_def.opset_import[0].version = 12 # Set an appropriate opset version

# Save the ONNX model
onnx.save(model_def, "model/simple_model.onnx")
print("Saved simple_model.onnx")
