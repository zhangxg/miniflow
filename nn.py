

from miniflow import *

if __name__ == "__main__":
  x, y = Input(), Input()

  f = Add(x, y)

  z = Input()

  # t = f + z
  t = Add(f, z)

  m = Add(f, x)

  zz = Add(t, m)

  feed_dict = {x: 10, y: 5, z:20}

  sorted_nodes = topological_sort(feed_dict)
  # output = forward_pass(f, sorted_nodes)
  output = forward_pass(zz, sorted_nodes)

  # NOTE: because topological_sort set the values for the `Input` nodes we could also access
  # the value for x with x.value (same goes for y).

  equation = ""
  parameters = []
  for k in feed_dict.keys():
    equation += "{}+"
    parameters.append(feed_dict[k])
  equation = equation[:-1] + " = {} "
  parameters.append(output)

  print(parameters)
  print((equation + "(according to miniflow)").format(*parameters))

