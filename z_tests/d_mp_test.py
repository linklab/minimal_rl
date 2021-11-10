import torch.multiprocessing as mp
from torch.multiprocessing import Process, cpu_count

from a_common.b_models import Policy


class Actor(Process):
    def __init__(self, actor_id, queue, policy):
        self.actor_id = actor_id
        self.queue = queue
        self.policy = policy
        super(Actor, self).__init__()

    def run(self):
        message = "I am here. This is actor #{0}".format(self.actor_id)
        self.queue.put(message)


class Learner(Process):
    def __init__(self, queue, n_actors, policy):
        super(Learner, self).__init__()
        self.queue = queue
        self.n_actors = n_actors
        self.policy = policy

    def run(self):
        n_received = 0
        while True:
            message = self.queue.get()
            n_received += 1

            print("[Number of Messages Received: {0:2}] MESSAGE: {1}".format(
                n_received, message
            ))

            if n_received >= self.n_actors:
                break


def main():
    queue = mp.SimpleQueue()
    policy = Policy(n_features=4, n_actions=3)

    n_cpu_cores = cpu_count()
    print("Number of CPU CORES: {0}".format(n_cpu_cores))
    n_actors = n_cpu_cores - 1

    actors = [Actor(actor_id, queue, policy) for actor_id in range(n_actors)]
    learner = Learner(queue, n_actors, policy)

    for actor in actors:
        actor.start()
    learner.start()

    for actor in actors:
        while actor.is_alive():
            actor.join(timeout=1)

    while learner.is_alive():
        learner.join(timeout=1)


if __name__ == "__main__":
    main()
