"""MLP Neural Architecture Search using PyTorch."""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from surfaces.modifiers import BaseModifier

from ..._base_nas import BaseNeuralArchitectureSearch


class MLPPyTorchNASFunction(BaseNeuralArchitectureSearch):
    """MLP Neural Architecture Search test function using PyTorch.

    **What is optimized:**
    This function optimizes the architecture of a Multi-Layer Perceptron (MLP)
    neural network. The search includes:
    - Number of hidden layers (1-3)
    - Number of units in each layer (32, 64, 128, 256)
    - Activation function (ReLU, Tanh, ELU)
    - Dropout rate (0.0, 0.1, 0.2, 0.3)

    **Search space structure:**
    The search space is conditional: only the first n_layers are active.
    For example, if n_layers=2, only layer_1_units and layer_2_units matter;
    layer_3_units is ignored.

    **What the score means:**
    The score is the validation accuracy (0.0 to 1.0) achieved by the network
    architecture on the MNIST digit classification task after training for
    a fixed number of epochs.

    **Optimization goal:**
    MAXIMIZE the validation accuracy. Higher scores indicate better-performing
    architectures. The goal is to find the optimal combination of network depth,
    width, activation function, and regularization (dropout) that achieves the
    best classification performance.

    **Computational cost:**
    Each evaluation trains a neural network for multiple epochs, making this
    an expensive function. Use small n_iter values for initial experiments.

    Parameters
    ----------
    n_epochs : int, default=5
        Number of training epochs per evaluation. Lower values speed up
        evaluation but may not allow the network to converge.
    batch_size : int, default=128
        Training batch size.
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer.
    subset_size : int, default=10000
        Number of training samples to use (MNIST has 60000 total).
        Smaller values speed up training for prototyping.
    n_jobs : int, default=2
        Number of CPU threads for PyTorch. Lower values reduce CPU load,
        keeping the system responsive during training.
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.

    Examples
    --------
    >>> from surfaces.test_functions.machine_learning import MLPPyTorchNASFunction
    >>> func = MLPPyTorchNASFunction(n_epochs=5, subset_size=5000)
    >>> func.search_space
    {'n_layers': [1, 2, 3], 'layer_1_units': [32, 64, 128, 256], ...}
    >>> result = func({"n_layers": 2, "layer_1_units": 128, "layer_2_units": 64,
    ...                "layer_3_units": 32, "dropout": 0.2, "activation": "relu"})
    >>> print(f"Validation accuracy: {result:.4f}")

    Notes
    -----
    Requires PyTorch and torchvision. Install with:
        pip install torch torchvision

    The function uses a subset of MNIST by default to keep evaluation time
    reasonable. For final benchmarking, increase subset_size and n_epochs.
    """

    name = "MLP PyTorch NAS"
    _name_ = "mlp_pytorch_nas"
    __name__ = "MLPPyTorchNASFunction"

    para_names = [
        "n_layers",
        "layer_1_units",
        "layer_2_units",
        "layer_3_units",
        "dropout",
        "activation",
    ]
    n_layers_default = [1, 2, 3]
    layer_1_units_default = [32, 64, 128, 256]
    layer_2_units_default = [32, 64, 128, 256]
    layer_3_units_default = [32, 64, 128, 256]
    dropout_default = [0.0, 0.1, 0.2, 0.3]
    activation_default = ["relu", "tanh", "elu"]

    def __init__(
        self,
        n_epochs: int = 5,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        subset_size: int = 10000,
        n_jobs: int = 2,
        objective: str = "maximize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        use_surrogate: bool = False,
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.subset_size = subset_size
        self.n_jobs = n_jobs

        super().__init__(
            objective=objective,
            modifiers=modifiers,
            memory=memory,
            collect_data=collect_data,
            callbacks=callbacks,
            catch_errors=catch_errors,
            use_surrogate=use_surrogate,
        )

    @property
    def search_space(self) -> Dict[str, Any]:
        """Search space for MLP architecture optimization."""
        return {
            "n_layers": self.n_layers_default,
            "layer_1_units": self.layer_1_units_default,
            "layer_2_units": self.layer_2_units_default,
            "layer_3_units": self.layer_3_units_default,
            "dropout": self.dropout_default,
            "activation": self.activation_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function for MLP NAS."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms

        # Limit CPU threads to keep system responsive
        torch.set_num_threads(self.n_jobs)

        # Load MNIST dataset
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        val_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

        # Use subset for faster evaluation
        if self.subset_size < len(train_dataset):
            train_indices = np.random.RandomState(42).choice(
                len(train_dataset), self.subset_size, replace=False
            )
            train_dataset = Subset(train_dataset, train_indices)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_epochs = self.n_epochs
        lr = self.learning_rate

        def objective_function(params: Dict[str, Any]) -> float:
            # Build network architecture
            n_layers = params["n_layers"]
            dropout_rate = params["dropout"]
            activation = params["activation"]

            # Activation function mapping
            activation_map = {
                "relu": nn.ReLU(),
                "tanh": nn.Tanh(),
                "elu": nn.ELU(),
            }
            act_fn = activation_map[activation]

            # Build sequential model
            layers = []
            input_dim = 28 * 28  # MNIST flattened

            for i in range(1, n_layers + 1):
                units = params[f"layer_{i}_units"]
                layers.append(nn.Linear(input_dim, units))
                layers.append(act_fn)
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                input_dim = units

            # Output layer
            layers.append(nn.Linear(input_dim, 10))

            model = nn.Sequential(*layers).to(device)

            # Train model
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            for epoch in range(n_epochs):
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                    data = data.view(data.size(0), -1).to(device)
                    target = target.to(device)

                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

            # Evaluate on validation set
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data = data.view(data.size(0), -1).to(device)
                    target = target.to(device)
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            accuracy = correct / total
            return accuracy

        self.pure_objective_function = objective_function
