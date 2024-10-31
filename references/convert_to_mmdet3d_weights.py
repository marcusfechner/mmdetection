import argparse

import torch


def convert_weights(input_path, output_path):
    # Load the .pth file
    input_file = open(input_path, 'rb')
    checkpoint = torch.load(input_file)
    
    # Extract the 'model' dictionary
    model_dict = checkpoint.get('model')
    
    if model_dict is None:
        raise ValueError("The input .pth file does not contain a 'model' key.")
    
    # Save the 'model' dictionary to a new .pth file
    
    torch.save(model_dict, output_path)
    print(f"Model weights saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .pth file to extract 'model' dictionary.")
    parser.add_argument('input_path', type=str, help="Path to the input .pth file")
    parser.add_argument('output_path', type=str, help="Path to save the output .pth file")
    
    args = parser.parse_args()
    
    convert_weights(args.input_path, args.output_path)