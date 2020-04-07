import torch, yaml

from core.models.CornerNet_Squeeze import model

net = model()
net = net.eval()

size = 511  # [255, 306, 357, 408, 459, 511]
input_shape = [1, 3, size, size]

with torch.no_grad():
    input_data = torch.randn(input_shape, requires_grad=False)
    scripted_net = torch.jit.trace(net, [input_data]).eval()
    scripted_net.save("cache/squeeze_{0}.pt".format(size))

ops = torch.jit.export_opnames(scripted_net)
with open("cache/model_ops.yaml", 'w') as output:
    yaml.dump(ops, output)

graph = scripted_net.graph.copy()
torch._C._jit_pass_inline(graph)
with open("cache/graph.txt", 'w') as f:
    f.write(str(graph))
