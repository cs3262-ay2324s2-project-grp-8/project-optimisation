import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class Brain(nn.Module):
    def __init__(self, state_size, action_size, brain_name, arguments):
        super(Brain, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.weight_backup = brain_name
        self.batch_size = arguments['batch_size']
        self.learning_rate = arguments['learning_rate']
        self.test = arguments['test']
        self.num_nodes = arguments['number_nodes']
        self.optimizer_model = arguments['optimizer']
        
        # Network for the primary model
        self.model = self._build_model()
        # Network for the target model
        self.model_ = self._build_model()
        
        # Choose optimizer for the primary model only
        if self.optimizer_model == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_model == 'RMSProp':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            print('Invalid optimizer!')

    def _build_model(self):
        # Defines the network structure
        model = nn.Sequential(
            nn.Linear(self.state_size, self.num_nodes),
            nn.ReLU(),
            nn.Linear(self.num_nodes, self.num_nodes),
            nn.ReLU(),
            nn.Linear(self.num_nodes, self.action_size)
        )
        
        if self.test:
            if os.path.isfile(self.weight_backup):
                self.load_state_dict(torch.load(self.weight_backup))
            else:
                print('Error: No such file')

        return model

    def forward(self, x, target=False):
        if target:
            # Pass through the target model
            return self.model_(x)
        else:
            # Pass through the primary model
            return self.model(x)

    def train_model(self, x, y, sample_weight=None, epochs=1, verbose=0):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        if sample_weight is not None:
            sample_weight_tensor = torch.tensor(sample_weight, dtype=torch.float32)
        else:
            # If no sample_weight is provided, use a tensor of ones (no weighting)
            sample_weight_tensor = torch.ones(x.shape[0], dtype=torch.float32)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.forward(x_tensor)
            loss = nn.SmoothL1Loss(outputs, y_tensor)
            
            # Apply sample weights
            weighted_loss = (loss * sample_weight_tensor).mean()
            
            weighted_loss.backward()
            self.optimizer.step()
            
            if verbose:
                print(f"Epoch {epoch}, Loss: {weighted_loss.item()}")

    def predict(self, state, target=False):
        with torch.no_grad():  # Ensure no gradients are computed
            state_tensor = torch.tensor(state, dtype=torch.float32)
            if target:
                # Use the target model for prediction
                self.model_.eval()  # Set the target model to evaluation mode
                prediction = self.model_(state_tensor)
                self.model_.train()  # Set the target model back to training mode
            else:
                # Use the primary model for prediction
                self.model.eval()  # Set the primary model to evaluation mode
                prediction = self.model(state_tensor)
                self.model.train()  # Set the primary model back to training mode
            return prediction.numpy()

    def predict_one_sample(self, state, target=False):
        return self.predict(state.reshape(1, self.state_size), target=target).flatten()

    def update_target_model(self):
        self.model_.load_state_dict(self.model.state_dict())

    def save_model(self):
        torch.save(self.state_dict(), self.weight_backup)
