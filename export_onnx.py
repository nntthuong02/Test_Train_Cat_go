import torch
import torch.onnx
import argparse
from model import GoNet

def export_to_onnx(pth_path="go5x5_model.pth", onnx_path="go5x5_model.onnx"):
    # Load model
    model = GoNet()
    try:
        model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"Loaded weights from {pth_path}")
    except FileNotFoundError:
        print(f"Warning: {pth_path} not found. Exporting model with random weights.")

    # Create dummy input based on (N, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 5, 5)

    # Export
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['policy', 'value'],
        dynamic_axes={'input': {0: 'batch_size'}, 'policy': {0: 'batch_size'}, 'value': {0: 'batch_size'}}
    )
    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_path", type=str, default="go5x5_model.pth", help="Path to the PyTorch model (.pth)")
    parser.add_argument("--onnx_path", type=str, default="go5x5_model.onnx", help="Path to save the ONNX model")
    args = parser.parse_args()
    
    export_to_onnx(pth_path=args.pth_path, onnx_path=args.onnx_path)
