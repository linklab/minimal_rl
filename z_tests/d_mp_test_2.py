from torch.multiprocessing import SimpleQueue, Process


def proc_a(queue):
    data = 1000
    queue.put(data)


def proc_b(queue):
    data = queue.get()
    print(data)


if __name__ == '__main__':
    yyy = "hi"
    print("%05s" % yyy)

    b = [(1, 2), (20, 30), (200, 300)]
    print(list(zip(*b)))

    queue = SimpleQueue()
    p1 = Process(target=proc_a, args=(queue,))
    p2 = Process(target=proc_b, args=(queue,))

    p1.start()
    p2.start()
    p1.join()
    p2.join()


