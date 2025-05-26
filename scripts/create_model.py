# scripts/create_simple_model.py
import onnx
from onnx import helper
from onnx import TensorProto
import os

def create_and_save_identity_model(model_path="model/simple_model.onnx"):
    """
    Creates a simple ONNX model that takes a 2D float tensor [None, 2]
    (batch_size, num_features=2) as input and returns it as output (Identity).
    Saves the model to the specified path.
    """
    # Define input: 'input_tensor', type FLOAT, shape [None, 2] (dynamic batch size, 2 features)
    X = helper.make_tensor_value_info('input_tensor', TensorProto.FLOAT, [None, 2])

    # Define output: 'output_tensor', type FLOAT, shape [None, 2]
    Y = helper.make_tensor_value_info('output_tensor', TensorProto.FLOAT, [None, 2])

    # Create an Identity node
    # Takes 'input_tensor' and produces 'output_tensor'
    identity_node = helper.make_node(
        'Identity',        # operator_type
        ['input_tensor'],  # inputs
        ['output_tensor'], # outputs
    )

    # Create the graph (model)
    # A graph contains a name, a list of nodes, and lists of inputs/outputs
    graph_def = helper.make_graph(
        [identity_node],
        'simple-identity-model', # name for the graph
        [X],                     # graph inputs
        [Y],                     # graph outputs
    )

    # Create the model
    # A model contains a graph and metadata like opset version and producer name
    model_def = helper.make_model(graph_def, producer_name='openenclave-poc-script')

    # Set the opset version.
    # ONNX Runtime supports a range of opsets. Version 12 is a reasonable choice for Identity.
    model_def.opset_import[0].version = 12

    # Ensure the directory for the model_path exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the ONNX model
    onnx.save(model_def, model_path)
    print(f"ONNX model saved to: {model_path}")
    print(f"  Input: name='{X.name}', type=FLOAT, shape={[d.dim_value if d.dim_value > 0 else d.dim_param for d in X.type.tensor_type.shape.dim]}")
    print(f"  Output: name='{Y.name}', type=FLOAT, shape={[d.dim_value if d.dim_value > 0 else d.dim_param for d in Y.type.tensor_type.shape.dim]}")


if __name__ == "__main__":
    # Default path relative to the project root if script is in openenclave_ml_poc_prod/scripts/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_model_path = os.path.join(project_root, "model", "simple_model.onnx")
    
    # You can change the path here if needed
    create_and_save_identity_model(model_path=default_model_path)

