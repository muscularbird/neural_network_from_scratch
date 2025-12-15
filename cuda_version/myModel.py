import ast
import cupy as cp
from Tensor import Tensor
from Layer import Linear
from Activation import ReLU, Sigmoid

class Model:
    def __init__(self, layers, lr, batch_size, epochs):
        self.layers = layers
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params
    
    def put_in_file(self, filename):
        with open(filename, 'w') as f:
            f.write("layers=\n")
            for layer in self.layers:
                f.write(self.__format_layer(layer) + '\n')
            f.write(f"lr={self.lr}\n")
            f.write(f"batch_size={self.batch_size}\n")
            f.write(f"epochs={self.epochs}\n")
    
    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            layer_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith("layers="):
                    continue
                elif line.startswith("lr="):
                    self.lr = float(line.split('=')[1])
                elif line.startswith("batch_size="):
                    self.batch_size = int(line.split('=')[1])
                elif line.startswith("epochs="):
                    self.epochs = int(line.split('=')[1])
                else:
                    layer_lines.append(line)
            
            for layer, line in zip(self.layers, layer_lines):
                if isinstance(layer, Linear):
                    prefix = 'Linear(W='
                    if not line.startswith(prefix) or ', b=' not in line or not line.endswith(')'):
                        raise ValueError(f"Invalid serialized Linear layer: {line}")
                    weights_str, bias_part = line[len(prefix):].rsplit(', b=', 1)
                    biases_str = bias_part[:-1]  # drop trailing ')'

                    # Detect truncated/ellipsized arrays (e.g. "..." in saved file)
                    if '...' in weights_str or 'Ellipsis' in weights_str:
                        raise ValueError(
                            "Model file appears truncated: weights contain ellipses '...'.\n"
                            "Re-save the model ensuring full numeric arrays (use tolist()/repr on arrays),\n"
                            "or store the model in a binary format (numpy.save / cupy.save)."
                        )

                    weights = cp.array(ast.literal_eval(weights_str))
                    biases = cp.array(ast.literal_eval(biases_str))

                    layer.W.data = weights
                    layer.b.data = biases
                elif hasattr(layer, 'load_from_string'):
                    layer.load_from_string(line)

    @classmethod
    def from_file(cls, filename):
        """Load a complete model from a .nn file, reconstructing layers from scratch."""
        with open(filename, 'r') as f:
            content = f.read()
        
        layers = []
        lr = 0.001
        batch_size = 32
        epochs = 50
        
        # Remove "layers=" prefix and split into sections
        if "layers=" in content:
            layers_section, rest = content.split("lr=", 1)
            layers_text = layers_section.replace("layers=", "").strip()
            
            # Parse hyperparameters
            for line in ("lr=" + rest).split('\n'):
                line = line.strip()
                if line.startswith("lr="):
                    lr = float(line.split('=')[1])
                elif line.startswith("batch_size="):
                    batch_size = int(line.split('=')[1])
                elif line.startswith("epochs="):
                    epochs = int(line.split('=')[1])
            
            # Parse layers - each layer entry ends with a newline followed by either another layer or end
            layer_entries = []
            current_entry = ""
            paren_depth = 0
            
            for char in layers_text:
                current_entry += char
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                    if paren_depth == 0 and current_entry.strip():
                        layer_entries.append(current_entry.strip())
                        current_entry = ""
                elif char == '\n' and paren_depth == 0:
                    if current_entry.strip() in ["ReLU", "Sigmoid"]:
                        layer_entries.append(current_entry.strip())
                        current_entry = ""
            
            # Process each layer entry
            for entry in layer_entries:
                entry = entry.strip()
                if entry.startswith("Linear(W="):
                    prefix = 'Linear(W='
                    weights_str, bias_part = entry[len(prefix):].rsplit(', b=', 1)
                    biases_str = bias_part[:-1]  # Remove trailing ')'
                    
                    if '...' in weights_str or 'Ellipsis' in weights_str:
                        raise ValueError(
                            "Model file appears truncated: weights contain ellipses '...'.\n"
                            "Re-save the model ensuring full numeric arrays (use tolist()/repr on arrays),\n"
                            "or store the model in a binary format (numpy.save / cupy.save)."
                        )

                    weights = cp.array(ast.literal_eval(weights_str))
                    biases = cp.array(ast.literal_eval(biases_str))
                    
                    in_features, out_features = weights.shape
                    layer = Linear(in_features, out_features)
                    layer.W.data = weights
                    layer.b.data = biases
                    layers.append(layer)
                elif entry == "ReLU":
                    layers.append(ReLU())
                elif entry == "Sigmoid":
                    layers.append(Sigmoid())
        
        return cls(layers, lr, batch_size, epochs)

    def __format_layer(self, layer):
        layer_type = layer.__class__.__name__
        if isinstance(layer, Linear):
            weights = layer.W.data.tolist()
            biases = layer.b.data.tolist()
            return f"{layer_type}(W={weights}, b={biases})"

        if hasattr(layer, 'parameters'):
            param_shapes = [param.data.shape for param in layer.parameters() if hasattr(param, 'data')]
            if param_shapes:
                shapes = ', '.join(str(shape) for shape in param_shapes)
                return f"{layer_type}(param_shapes=[{shapes}])"

        return layer_type