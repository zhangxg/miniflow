
import numpy as np


class Node(object):

  def __init__(self, inbound_nodes=[]):
    self.inbound_nodes = inbound_nodes

    self.outbound_nodes = []
    for n in inbound_nodes:
      n.outbound_nodes.append(self)

    self.value = None

  def forward(self):
    raise NotImplementedError


class Input(Node):

  def __init__(self):
    Node.__init__(self)

  def forward(self, value=None):
    if value is not None:
      self.value = value


class Add(Node):

  def __init__(self, inbound_nodes):
    Node.__init__(self, inbound_nodes)

  def forward(self):
    self.value = sum([n.value for n in self.inbound_nodes])


class Multiple(Node):

  def __init__(self, inbound_nodes):
    Node.__init__(self, inbound_nodes)
    self.value = 1

  def forward(self):
    for n in self.inbound_nodes:
      self.value *= n.value


class Linear(Node):

  def __init__(self, inputs, weights, bias):
    # ## my work vs the tutorial's
    # Node.__init__(self, [inputs])
    # self.weights = weights
    # self.bias = bias
    # self.value = 0

    # the input, weights and bias are treated as inbound_nodes,
    # this is more clear
    Node.__init__(self, [inputs, weights, bias])

  def forward(self):
    # self.value = np.dot(self.inputs, self.weights) + self.bias

    # this works, but not quite intuitively.
    # self.value = np.dot(self.inbound_nodes[0].value, self.weights.value) + self.bias.value

    # the calculation is adjusted accordingly.
    self.value = np.dot(self.inbound_nodes[0].value, self.inbound_nodes[1].value) \
                 + self.inbound_nodes[2].value


class MatrixLinear(Node):

  def __init__(self, X, W, b):
    Node.__init__(self, [X, W, b])

  def forward(self):
    self.value = np.dot(self.inbound_nodes[0].value, self.inbound_nodes[1].value) \
                 + self.inbound_nodes[2].avlue


class Sigmoid(Node):

  def __init__(self, inputs):
    Node.__init__(self, [inputs])

  def forward(self):
    x = -self.inbound_nodes[0].value
    self.value = 1 / (1 + np.exp(x))


class MSE(Node):

  def __init__(self, a, y):
    Node.__init__(self, [a, y])

  def forward(self):
    y = self.inbound_nodes[1].value
    y_hat = self.inbound_nodes[0].value
    # self.value = np.sum(np.square(y - y_hat)) / len(y)
    self.value = np.mean(np.square(y - y_hat))
