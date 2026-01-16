from typing_extensions import TypedDict


class LayerConfig(TypedDict, closed=True):
    # Maximum number of hidden layers
    max_layers: int
    # Maximum number of nodes per hidden layer
    max_nodes_per_layer: int


class NetworkConfig(TypedDict, closed=True):
    #: Layers of the neural network
    layers: LayerConfig | list[int]
    #: Activation function used in hidden layers
    activation: list[str] | str
    #: Probability of dropout for dropout layers
    dropout: tuple[float, float] | float


class OptimizerConfig(TypedDict, closed=True):
    #: Optimizer used to train neural network
    name: list[str] | str
    #: Learning rate used to train neural network
    lr: tuple[float, float] | float
    #: Learning rate momentum used to train neural network
    momentum: tuple[float, float] | float
    #: Scheduler applied to learning rate
    scheduler: list[str] | str
    #: Decay used by learning rate scheduler
    decay: tuple[float, float] | float
