

from miniflow import *
from helper import *


def simple_add():
  x, y = Input(), Input()
  z = Input()

  f = Add([x, y, z])
  t = Add([f, z])

  m = Multiple([t, x, z])

  feed_dict = {x: 10, y: 5, z: 20}

  sorted_nodes = topological_sort(feed_dict)
  output = forward_pass(m, sorted_nodes)

  # NOTE: because topological_sort set the values for the `Input` nodes we could also access
  # the value for x with x.value (same goes for y).

  # fixme, this still doesn't reflect the graph
  equation = ""
  parameters = []
  for k in feed_dict.keys():
    equation += "{}+"
    parameters.append(feed_dict[k])
  equation = equation[:-1] + " = {} "
  parameters.append(output)

  print(parameters)
  print((equation + "(according to miniflow)").format(*parameters))


def linear():
  inputs, weights, bias = Input(), Input(), Input()

  f = Linear(inputs, weights, bias)

  feed_dict = {
    inputs: [6, 14, 3],
    weights: [0.5, 0.25, 1.4],
    bias: 2
  }

  graph = topological_sort(feed_dict)
  output = forward_pass(f, graph)

  print(output)  # should be 12.7 with this example


if __name__ == "__main__":
  # simple_add()
  linear()

