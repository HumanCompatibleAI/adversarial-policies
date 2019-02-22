from multiprocessing import Process, Pipe, Value


def value_test(value):
    print(value)
    value += 1


remotes, work_remotes = zip(*[Pipe() for _ in range(2)])
processes = [Process(target=value_test, args=())]

