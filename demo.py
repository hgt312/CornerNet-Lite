import torch, yaml

from core.models.CornerNet_Squeeze import model

net = model()
net = net.eval()

size = 511  # [255, 306, 357, 408, 459, 511]
input_shape = [1, 3, size, size]
input_data = torch.randn(input_shape)
scripted_net = torch.jit.trace(net, [input_data]).eval()
scripted_net.save("cache/squeeze_{0}.pt".format(size))

ops = torch.jit.export_opnames(scripted_net)
with open("cache/model_ops.yaml", 'w') as output:
    yaml.dump(ops, output)
