

# from miniflow import *
from helper import *
import miniflow_with_bp as mbp
from miniflow_with_bp import *


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


def matrix_linear():
  X, W, b = Input(), Input(), Input()

  f = Linear(X, W, b)

  X_ = np.array([[-1., -2.], [-1, -2], [-1, -2]])
  W_ = np.array([[2., -3, -1], [2., -3, -1]])
  b_ = np.array([-3., -5, -3])

  feed_dict = {X: X_, W: W_, b: b_}

  graph = topological_sort(feed_dict)
  output = forward_pass(f, graph)

  """
  Output should be:
  [[-9., 4.],
  [-9., 4.]]
  """
  print(output)


def linear_with_sigmoid():
  X, W, b = Input(), Input(), Input()

  f = Linear(X, W, b)
  g = Sigmoid(f)

  X_ = np.array([[-1., -2.], [-1, -2]])
  W_ = np.array([[2., -3], [2., -3]])
  b_ = np.array([-3., -5])

  feed_dict = {X: X_, W: W_, b: b_}

  graph = topological_sort(feed_dict)
  output = forward_pass(g, graph)

  """
  Output should be:
  [[  1.23394576e-04   9.82013790e-01]
   [  1.23394576e-04   9.82013790e-01]]
  """
  print(output)


def mse():
  y, a = Input(), Input()
  cost = MSE(y, a)

  y_ = np.array([1, 2, 3])
  a_ = np.array([4.5, 5, 10])

  feed_dict = {y: y_, a: a_}
  graph = topological_sort(feed_dict)
  # forward pass
  forward_pass(cost, graph)

  """
  Expected output

  23.4166666667
  """
  print(cost.value)


def bp():
  X, W, b = Input(), Input(), Input()
  y = Input()
  f = Linear(X, W, b)
  a = Sigmoid(f)
  cost = MSE(y, a)

  X_ = np.array([[-1., -2.], [-1, -2]])
  W_ = np.array([[2.], [3.]])
  b_ = np.array([-3.])
  y_ = np.array([1, 2])

  feed_dict = {
    X: X_,
    y: y_,
    W: W_,
    b: b_,
  }

  graph = topological_sort(feed_dict)
  forward_and_backward(graph)
  # return the gradients for each Input
  gradients = [t.gradients[t] for t in [X, y, W, b]]

  """
  Expected output

  [array([[ -3.34017280e-05,  -5.01025919e-05],
         [ -6.68040138e-05,  -1.00206021e-04]]), array([[ 0.9999833],
         [ 1.9999833]]), array([[  5.01028709e-05],
         [  1.00205742e-04]]), array([ -5.01028709e-05])]
  """
  print(gradients)


if __name__ == "__main__":
  # simple_add()
  # linear()
  # matrix_linear()
  # linear_with_sigmoid()
  # mse()
  bp()