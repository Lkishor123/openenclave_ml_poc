# scripts/create_model.py
import onnx
from onnx import helper
from onnx import TensorProto
import os

def create_and_save_identity_model(model_path="model/simple_model.onnx"):
    """
    Creates a simple ONNX model that takes a 2D float tensor [None, 2]
    (batch_size, num_features=2) as input and returns it as output (Identity).
    Saves the model to the specified path, targeting a specific opset and IR version
    for broader compatibility.
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
    graph_def = helper.make_graph(
        [identity_node],
        'simple-identity-model', # name for the graph
        [X],                     # graph inputs
        [Y],                     # graph outputs
    )

    # Create the model
    # Explicitly set the ir_version. Your ONNX Runtime supports up to IR version 10.
    # Common ONNX IR versions:
    # IRv7: ONNX 1.6
    # IRv8: ONNX 1.7-1.9
    # IRv9: ONNX 1.10-1.12
    # IRv10: ONNX 1.13-1.16
    # Your onnx python library (1.18.0) likely defaults to IRv11.
    model_def = helper.make_model(graph_def, producer_name='openenclave-poc-script', ir_version=10) # Explicitly set IR version

    # Set the opset version.
    # Opset 10 is compatible with IR version 10.
    model_def.opset_import[0].version = 10 


    # Ensure the directory for the model_path exists
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the model_path relative to the script's parent directory
    # (assuming script is in 'scripts/' and model should be in 'model/')
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(script_dir), model_path)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the ONNX model
    onnx.save(model_def, model_path)
    print(f"ONNX model saved to: {model_path}")
    print(f"  Opset version: {model_def.opset_import[0].version}")
    print(f"  IR version: {model_def.ir_version}") # Print the IR version for confirmation
    print(f"  Input: name='{X.name}', type=FLOAT, shape={[d.dim_value if d.dim_value > 0 else d.dim_param for d in X.type.tensor_type.shape.dim]}")
    print(f"  Output: name='{Y.name}', type=FLOAT, shape={[d.dim_value if d.dim_value > 0 else d.dim_param for d in Y.type.tensor_type.shape.dim]}")


if __name__ == "__main__":
    # Default path relative to the project root if script is in openenclave_ml_poc/scripts/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_model_path = os.path.join(project_root, "model", "simple_model.onnx")
    
    create_and_save_identity_model(model_path=default_model_path)
