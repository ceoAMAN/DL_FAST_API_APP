import os
import torch
import torch.nn as nn
import torch.optim as optim
from app.model import SimpleModel

def train_dummy_model():
    model = SimpleModel()
    
   
    X = torch.randn(100, 4)
    y = torch.randint(0, 3, (100,))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
   
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")
    
    
    os.makedirs("model", exist_ok=True)
    
    
    torch.save(model.state_dict(), "model/model.pth")
    print("Model saved to model/model.pth")

if __name__ == "__main__":
    train_dummy_model()