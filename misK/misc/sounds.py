import time

from progress import progress_bar

DURATION = 0.1


def signal(duration, symbol):
    time.sleep(duration)
    print(symbol, end='', flush=True)


dot = lambda: signal(DURATION, '·\a')
dash = lambda: signal(3 * DURATION, '−\a')
symbol_space = lambda: signal(DURATION, '')
letter_space = lambda: signal(3 * DURATION, '')
word_space = lambda: signal(7 * DURATION, ' ')

if __name__ == "__main__":
    n = 100000
    for i in range(n):
        progress_bar(i, n)

    while True:
        dot()
        symbol_space()
        dot()
        symbol_space()
        dot()
        letter_space()
        dash()
        symbol_space()
        dash()
        symbol_space()
        dash()
        letter_space()
        dot()
        symbol_space()
        dot()
        symbol_space()
        dot()
        word_space()

