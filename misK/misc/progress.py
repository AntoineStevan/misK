# import time
#
# bars = "\\|/-"
#
# bar = 0
# while True:
#     print(bars[bar], end='\r')
#     bar = (bar + 1) % len(bars)
#     time.sleep(0.5)
#
#
# from itertools import cycle
# from time import sleep
#
# for frame in cycle(r'-\|/-\|/'):
#     print('\r', frame, sep='', end='', flush=True)
#     sleep(0.2)


def progress_bar(current, total, width=100):
    progress = 100 * current // (total-1)
    done = progress * width // 100
    print('\r[', '#' * done, ' ' * (width - done), ']', f" {progress:.0f}%", sep='', end='')


if __name__ == "__main__":
    n = 100000
    for i in range(n):
        progress_bar(i, n)
