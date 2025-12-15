from myModel import Model
from Activation import ReLU, Sigmoid
from Layer import Linear

model = Model([
    Linear(64, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 64),
    Sigmoid(),
    Linear(64, 5),

], lr=0.0001, batch_size=64, epochs=100)