from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import lzma
import flwr as fl
import os
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from sklearn.metrics import precision_score, recall_score

from typing import Callable, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import time
import polyline 
from polyline import decode
from torchvision.datasets import ImageFolder

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

NUM_CLIENTS = 10
def _power_law_split(
    sorted_trainset: Dataset,
    num_partitions: int,
    num_labels_per_partition: int = 2,
    min_data_per_partition: int = 10,
    mean: float = 0.0,
    sigma: float = 2.0,
) -> Dataset:
    """Partition the dataset following a power-law distribution. It follows the.

    implementation of Li et al 2020: https://arxiv.org/abs/1812.06127 with default
    values set accordingly.
    """
    targets = sorted_trainset.targets
    full_idx = list(range(len(targets)))

    class_counts = np.bincount(sorted_trainset.targets)
    labels_cs = np.cumsum(class_counts)
    labels_cs = [0] + labels_cs[:-1].tolist()

    partitions_idx: List[List[int]] = []
    num_classes = len(np.bincount(targets))
    hist = np.zeros(num_classes, dtype=np.int32)

    # assign min_data_per_partition
    min_data_per_class = int(min_data_per_partition / num_labels_per_partition)
    for u_id in range(num_partitions):
        partitions_idx.append([])
        for cls_idx in range(num_labels_per_partition):
            # label for the u_id-th client
            cls = (u_id + cls_idx) % num_classes
            # record minimum data
            indices = list(
                full_idx[
                    labels_cs[cls]
                    + hist[cls] : labels_cs[cls]
                    + hist[cls]
                    + min_data_per_class
                ]
            )
            partitions_idx[-1].extend(indices)
            hist[cls] += min_data_per_class

    # add remaining images following power-law
    probs = np.random.lognormal(
        mean,
        sigma,
        (num_classes, int(num_partitions / num_classes), num_labels_per_partition),
    )
    remaining_per_class = class_counts - hist
    # obtain how many samples each partition should be assigned for each of the
    # labels it contains
    # pylint: disable=too-many-function-args
    probs = (
        remaining_per_class.reshape(-1, 1, 1)
        * probs
        / np.sum(probs, (1, 2), keepdims=True)
    )

    for u_id in range(num_partitions):
        for cls_idx in range(num_labels_per_partition):
            cls = (u_id + cls_idx) % num_classes
            count = int(probs[cls, u_id // num_classes, cls_idx])

            # add count of specific class to partition
            indices = full_idx[
                labels_cs[cls] + hist[cls] : labels_cs[cls] + hist[cls] + count
            ]
            partitions_idx[u_id].extend(indices)
            hist[cls] += count

    # construct subsets
    partitions = [Subset(sorted_trainset, p) for p in partitions_idx]
    return partitions

#IID Dataset partition

def load_datasets(num_clients: int):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),  # Resize all images to 512x512
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ]
    )
    trainset = ImageFolder(root="/content/Lung Disease Dataset/train", transform=transform)
    testset = ImageFolder(root="/content/Lung Disease Dataset/test", transform=transform)
    val= ImageFolder(root="/content/Lung Disease Dataset/val", transform=transform)
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // num_clients
    remaining_samples = len(trainset) - partition_size * num_clients

    datasets = _power_law_split(trainset, num_partitions=num_clients)
    trainloaders = []
    valloaders = []
    # DataLoader for testset
    testloader = DataLoader(testset, batch_size=32)
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))
    testloader = DataLoader(testset, batch_size=32)
    return trainloaders, valloaders, testloader


trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS)

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # Adjusted input size based on the output size after convolutional layers
        self.fc1 = nn.Linear(16 * 125 * 125, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        #print("After conv1 and maxpool:", x.size())
        x = self.pool(F.relu(self.conv2(x)))
        #print("After conv2 and maxpool:", x.size())
        # Adjusted based on the output size from conv2
        x = x.view(-1, 16 * 125 * 125)
        #print("After view:", x.size())
        x = F.relu(self.fc1(x))
        #print("After fc1:", x.size())
        x = F.relu(self.fc2(x))
        #print("After fc2:", x.size())
        x = self.fc3(x)
        #print("After fc3:", x.size())
        return x


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss = criterion(net(images), labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}
params = get_parameters(Net())

def encode_parameters(params):
    flat_params = np.concatenate([p.flatten() for p in params])
    int_params = (flat_params * 1e6).astype(int)
    encoded = polyline.encode([(v, 0) for v in int_params])

    # Log the encoded data for debugging
    print("Encoded Parameters:", encoded)
    return encoded
    
def decode_parameters(encoded, shapes):
    decoded = polyline.decode(encoded)
    deltas = np.array([v[0] for v in decoded])
    int_params = np.cumsum(deltas) / 1e6  # Reverse scaling
    split_indices = np.cumsum([np.prod(shape) for shape in shapes])[:-1]
    param_tensors = [torch.tensor(arr.reshape(shape)) for arr, shape in zip(np.split(int_params, split_indices), shapes)]
    return param_tensors

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self,config):
      params = [p.detach().cpu().numpy() for p in self.model.parameters()]
      encoded = encode_parameters(params)
      return [encoded]  # Wrap it in a list


    def set_parameters(self, parameters):
      encoded = parameters[0]  # Extract first element if passed as list
      shapes = [p.shape for p in self.model.parameters()]
      decoded_params = decode_parameters(encoded, shapes)
      for param, new_param in zip(self.model.parameters(), decoded_params):
          param.data = new_param.float()

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        num_labels = 0
        for _, labels in self.trainloader:
            num_labels += len(labels)
        data_label_ratio = num_labels / len(self.trainloader.dataset)

        # Add timestamp and data_label_ratio to metrics
        metrics = {"timestamp": time.time(), "data_label_ratio": data_label_ratio}

        return get_parameters(self.net), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def test_with_metrics(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    loss_fn = nn.CrossEntropyLoss()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')

    return total_loss, accuracy, precision, recall

def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    start_time = time.time()  # Record start time
    net = Net().to(DEVICE)
    valloader = valloaders[0]
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy, precision, recall = test_with_metrics(net, valloader)
    end_time = time.time()  # Record end time
    round_time = end_time - start_time  # Calculate round time
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy} / precision {precision} / recall {recall}")
    print(f"Time taken for round {server_round}: {round_time} seconds")
    return loss, {"accuracy": accuracy, "precision": precision, "recall": recall}


class FceFed(FedAvg):
    def __init__(
        self,
        fraction_fit: float = 0.3,
        fraction_evaluate: float = 0.3, #Sample 50% of available clients for evaluation
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 1, #Allow evaluation with as few as 1 client
        min_available_clients: int = 2,

    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate
        self.initial_lr = 0.001
        self.mu = 0.1
        self.decay_factor = 0.01
    def __repr__(self) -> str:
        return "FceFed"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        net = Net()
        ndarrays = get_parameters(net)
        return fl.common.ndarrays_to_parameters(ndarrays)



    def aggregate_fit(self, server_round, results, failures):
        current_timestamp = time.time()
        weights_results = []
        for client, fit_res in results:
            parameters = parameters_to_ndarrays(fit_res.parameters)
            client_timestamp = fit_res.metrics["timestamp"]
            staleness = current_timestamp - client_timestamp

            eta_k = self.initial_lr * np.exp(-self.decay_factor * staleness)
            fairness_factor = max(0, 1 - self.mu * staleness)
            data_ratio = fit_res.metrics["data_label_ratio"]
            client_weight = eta_k * data_ratio * fairness_factor


        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated

def client_fn(cid) -> FlowerClient:
    net = Net().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=200),  
    strategy=FceFed(),
    client_resources=client_resources,
)