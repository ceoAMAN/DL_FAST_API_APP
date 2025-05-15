import torch
from .model import SimpleModel
from .utils import get_model_path

class ModelInference:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        model_path = get_model_path()
        model = SimpleModel()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        self.model = model
    
    def predict(self, features):
        with torch.no_grad():
            inputs = torch.tensor([features], dtype=torch.float32)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            return predicted.item()

inference = ModelInference()