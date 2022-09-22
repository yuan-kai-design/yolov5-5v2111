import torch

a = torch.tensor([1, 2, 3.], requires_grad=True)
print(a.grad)  # None
out = a.sigmoid()
print(out)  # tensor([0.7311, 0.8808, 0.9526], grad_fn=<SigmoidBackward>)

# 添加detach(),c的requires_grad为False
c = out.detach()
print(c)  # tensor([0.7311, 0.8808, 0.9526])
print(c.grad_fn,c.requires_grad) # None False
c.requires_grad=True
print(out)  # tensor([0.7311, 0.8808, 0.9526], grad_fn=<SigmoidBackward>)与上面的out=a.sigmoid()对应

# 这时候没有对c进行更改，所以并不会影响backward()
out.sum().backward()
print(a.grad)  # tensor([0.1966, 0.1050, 0.0452])
