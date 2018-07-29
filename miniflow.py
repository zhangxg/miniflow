
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
    Node.__init__(self, [inputs])
    self.weights = weights
    self.bias = bias
    self.value = 0

  def forward(self):
    # self.value = np.dot(self.inputs, self.weights) + self.bias
    self.value = np.dot(self.inbound_nodes[0].value, self.weights.value) + self.bias.value




