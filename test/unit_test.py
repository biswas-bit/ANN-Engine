import numpy as np
from ann_engine.layers import Sequential, Dense
from ann_engine.losses import CrossEntropyLoss
from ann_engine.optimizers import Adam

# creating a simple neural network
model = Sequential([
    Dense(128, activation='relu',input_dim=784),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

#compile the model with loss and optimizer
model.compile(loss = 'cross_entropy', optimizer=Adam(learning_rate=0.001))
model.summary()
input()