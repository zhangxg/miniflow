

## todo: want to generate: "{}+{}+{}".format([from_a_list])

feed_dict = {"x": 10, "y": 5, "z": 20}

ss = ""
for k in feed_dict.keys():
  ss += "{}+"

  # print("{}+".join([str(i) for i in range(len(feed_dict.keys()))]))
  # print(k, feed_dict[k])

# print([feed_dict[k] for k in feed_dict.keys()])
# print(ss[:-1])
# print(tuple([feed_dict[k] for k in feed_dict.keys()]))
# print(ss[:-1].format((tuple([feed_dict[k] for k in feed_dict.keys()]))))
print(",".join([str(feed_dict[k] ) for k in feed_dict.keys()]))

print("{}+{}+{}".format(",".join(str([feed_dict[k] for k in feed_dict.keys()]))))
# "{}+{}+{}".format(10, 5, 20)
print(*[feed_dict[k] for k in feed_dict.keys()])
# print("".join(*[feed_dict[k] for k in feed_dict.keys()]))
