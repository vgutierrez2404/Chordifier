import gradio as gr
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, encoded_dim):
        super(Encoder, self).__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Sequential(
                nn.Linear(input_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, encoded_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Decoder(nn.Module):
    def __init__(self, encoded_dim, hidden_dims, output_dim):
        super(Decoder, self).__init__()
        layers = []
        hidden_dims.reverse()
        for h_dim in hidden_dims:
            layers.append(nn.Sequential(
                nn.Linear(encoded_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ))
            encoded_dim = h_dim
        layers.append(nn.Linear(encoded_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim=12, hidden_dims=None, encoded_dim=12, learning_rate=1e-4):
        super(Autoencoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        self.encoder = Encoder(input_dim, hidden_dims, encoded_dim)
        self.decoder = Decoder(encoded_dim, hidden_dims, input_dim)
        self.learning_rate = learning_rate

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        print('llego')
        inputs = batch
        inputs = inputs.view(inputs.size(0), -1)
        outputs = self.forward(inputs)
        loss = nn.MSELoss()(outputs, inputs)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch
        inputs = inputs.view(inputs.size(0), -1)
        outputs = self.forward(inputs)
        loss = nn.MSELoss()(outputs, inputs)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def encode(self, x): 
        return self.encoder(x)

# Load the trained model
model = Autoencoder(input_dim=12, hidden_dims=[128, 64, 32], encoded_dim=12)
model.load_state_dict(torch.load("autoencoder.pth"))
model.eval()

# Define a function to make predictions using the model
def predict(input_array):
    input_tensor = torch.tensor(input_array, dtype=torch.float32)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    return output_tensor.numpy()

# Define Gradio interface
input_component = gr.inputs.Dataframe(headers=["Feature 1", "Feature 2", "...", "Feature N"], dtype=float)
output_component = gr.outputs.Dataframe(headers=["Feature 1", "Feature 2", "...", "Feature N"], dtype=float)

gr_interface = gr.Interface(fn=predict, inputs=input_component, outputs=output_component, live=True)

# Launch the Gradio app
if __name__ == "__main__":
    gr_interface.launch()