import torch.multiprocessing as mp
a = [mp.Pipe() for _ in range(3)]
print(*a)

b = [(1, 2), (20, 30), (200, 300)]
print(list(zip(*b)))