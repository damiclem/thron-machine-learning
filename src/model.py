# Dependencies
import torch.nn as nn
import torch


# Define neural network
class MultiLabelClassifier(nn.Module):

    # Constructor
    def __init__(self, out_features, verbose=False):
        # Call parent constructor
        super().__init__()
        # Retrieve default MobileNetV2 architecture
        self.out = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True, verbose=verbose)
        # Define number of input features for the last layer
        in_features = self.out.classifier[1].in_features
        # Update latest layer
        self.out.classifier = nn.Sequential(
            # Use dropout, as in standard architecture
            nn.Dropout(p=0.2),
            # Then, use linear layer as last
            nn.Linear(in_features=in_features, out_features=out_features)
        )
        # Define sigmoid layer
        self.sigmoid = nn.Sigmoid()

    # Predict
    def forward(self, x):
        # Pass prediction output to sigmoid layer
        return self.sigmoid(self.out(x))

    @property
    def device(self):
        return next(self.parameters()).device


def train(model, data, optimizer, loss_fn):
    """
    Train the given model on all the partitions of the given data loader.

    :param model: Given model to train
    :param data: Target data to train model
    :param optimizer: Chosen optimizer object
    :param loss_fn: Chosen loss function
    :return: Iterator for train loss
    """
    # Set model in training model
    model.train()
    # Define device
    device = model.device
    # Loop through each batch in model
    for sample in data:
        # Retrieve images and tokens
        images, tokens = sample['image'], sample['tokens']
        # Move images and tokens on the correct device
        images, tokens = images.to(device), tokens.to(device)
        # Reset optimizer
        optimizer.zero_grad()
        # Compute results
        predicted = model(images)
        # Compute loss
        loss = loss_fn(predicted, tokens.type(torch.float))
        # Update parameters
        loss.backward()
        optimizer.step()
        # Yield batch loss
        yield loss.item()


def test(model, data, loss_fn, eval_fn):
    """

    :param model: Model to be tested
    :param data: Test dataset iterator
    :param loss_fn: Loss function to compute test loss
    :param eval_fn: Evaluation function to compute other metrics
    :return: Iterator for test loss and evaluation
    """
    # Set model in test mode
    model.eval()
    # Define model device
    device = model.device
    # Define default device
    cpu = torch.device('cpu')
    # Disable gradient computation
    with torch.no_grad():
        # Loop through each batch in test dataset
        for batch in data:
            # Retrieve input images and true tokens
            images, tokens = batch['image'], batch['tokens']
            # Move both images and tokens on device
            images, tokens = images.to(device), tokens.type(torch.float).to(device)
            # Make predictions
            predicted = model(images)
            # Evaluate loss
            loss = loss_fn(predicted, tokens).item()
            eval = eval_fn(predicted.to(cpu), tokens.to(cpu))
        # Yield results of both loss and evaluation function
        yield loss, eval


def save(model, path):
    """

    :param model: Model whose parameters must be stored
    :param path: Path where to store model parameter
    :return: Does not return nothing
    """
    torch.save(model.state_dict(), path)


# Main
if __name__ == '__main__':
    # Not implemented
    raise NotImplementedError
