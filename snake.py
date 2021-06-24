import curses
import time


def main(screen):
    # initializations.
    curses.curs_set(0)  # hide the cursor.
    screen.nodelay(True)  # do not block I/O calls.
    snake = [(0, i) for i in reversed(range(20))]
    directions = {
        curses.KEY_UP: (-1, 0),
        curses.KEY_DOWN: (1, 0),
        curses.KEY_LEFT: (0, -1),
        curses.KEY_RIGHT: (0, 1)
    }
    direction = directions[curses.KEY_RIGHT]

    while True:
        # draw the snake.
        screen.erase()
        screen.addstr(*snake[0], '@')
        for segment in snake[1:]:
            screen.addstr(*segment, '*')

        # update the snake.
        direction = directions.get(screen.getch(), direction)
        snake.pop()
        new_pos = tuple(map(sum, zip(snake[0], direction)))
        snake.insert(0, new_pos)

        # refresh and pause.
        screen.refresh()
        time.sleep(33 / 1000)


if __name__ == "__main__":
    curses.wrapper(main)
