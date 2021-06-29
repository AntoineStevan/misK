import time


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

class ProgressBar:
    def __init__(self, total, width=100):
        self.total = total
        self.progress = 0
        self.width = width
        self.formats = ["{:5.1f}%",
                        "{:" + str(len(str(self.total))) + "d}/{}",
                        "[{:02d}:{:02d}<{:02d}:{:02d}, {:4.2f} {}]"]

        self.start = time.time()
        self.times = {}

    def _bar(self):
        progress = 100 * self.progress / self.total
        bar_format = ' '.join([self.formats[0], self._inner_bar(progress), self.formats[1], self.formats[2]])
        t = time.time() - self.start
        self.times[self.progress] = t
        pred = int(t * self.total / self.progress - t)
        t = int(t)
        speed = list(self.times.values())[-10:]
        speed = (speed[-1] - speed[0]) / len(speed)
        speed, it = (speed, "its/s") if speed >= 1 else (1/speed, "s/it") if speed > 0 else (float("nan"), "----")
        return bar_format.format(progress, self.progress, self.total,
                                 t // 60, t % 60, pred // 60, pred % 60,
                                 speed, it)

    def __call__(self, incr=1, force=None):
        if force is None:
            self.progress += incr
        else:
            self.progress = force

        print('\r', self._bar(), sep='', end='')

    def _inner_bar(self, progress):
        done = int(progress * self.width / 100)
        digit = int(round((progress - int(progress)) * 10, 0))
        bar = '#' * done + str('#' if digit == 10 else digit) + ' ' * (self.width - done - 1)
        return '[' + bar[:self.width] + ']'

    def close(self):
        print('\r', self._bar(), sep='')


if __name__ == "__main__":
    from tqdm import tqdm, trange
    from time import sleep

    # with tqdm(total=100, ascii=True) as pbar:
    #     for i in range(100):
    #         sleep(0.1)
    #         pbar.update(1)

    # for i in trange(3, desc='1st loop'):
    #     for j in tqdm(range(100), desc='2nd loop', leave=False):
    #         sleep(0.01)

    n = 5123
    bar = ProgressBar(n, width=100)
    for i in range(n):
        sleep(0.1)
        bar(incr=1)
    bar.close()
