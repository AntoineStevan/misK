import traceback
from datetime import datetime

FILE = None
FUNC = None
MOTHER_FUNCTION = None
FUNCTION_CHANGE = False

VERBOSE = False


def _get_head(depth=1):
    global FUNC
    stack = traceback.format_stack()
    file, line, FUNC = stack[-2 - depth].split('\n')[0].split(", ")
    file = file.split('"')[1]
    line = line.split(' ')[1]
    FUNC = FUNC.split(' ')[1]
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S,%f")

    return f"[{now}][{file}][{FUNC}][{line}][LOG] "


def open_logger(filename, mode='a', verbose=False):
    global FILE, VERBOSE
    FILE = open(filename, mode=mode)
    FILE.write(_get_head() + "logger opened." + '\n')
    VERBOSE = verbose


def log(*args, prt=True, log=True, **kwargs):
    global VERBOSE
    if VERBOSE:
        global FUNCTION_CHANGE, MOTHER_FUNCTION, FILE, FUNC

        head = _get_head()

        FUNCTION_CHANGE = FUNC != MOTHER_FUNCTION
        MOTHER_FUNCTION = FUNC

        if prt:
            if FUNCTION_CHANGE:
                print()
            print("[@]", *args, **kwargs)
        if log:
            if FUNCTION_CHANGE:
                FILE.write('\n')
            FILE.write(head + ' '.join(map(str, args)) + '\n')


def void(*args, **kwargs):
    pass


def close_logger():
    global FUNCTION_CHANGE, MOTHER_FUNCTION, FILE, FUNC, VERBOSE
    if FILE is not None and not FILE.closed:

        head = _get_head()

        FUNCTION_CHANGE = FUNC != MOTHER_FUNCTION
        MOTHER_FUNCTION = FUNC

        if FUNCTION_CHANGE:
            print()
            FILE.write('\n')
        FILE.write(head + "logger closed." + '\n')
        FILE.write('-'*100 + "\n")
        FILE.write('-'*100 + "\n")
        FILE.close()

    VERBOSE = False


def sub_main(log=print):
    log("sub_main 1")
    log("sub_main 2")
    log("sub_main 3")
    dico = dict([(chr(97 + i), sum([10 ** j for j in range(i)])) for i in range(20)])
    log("dictionary example:", dico)
    log("sub_main end")


def main(log=print):
    log("first line")
    for line in range(10, 20, 3):
        log(f"line n° {line}")
    sub_main(log=log)
    log("end")


if __name__ == "__main__":
    open_logger("log/log.log", verbose=True)
    main(log=log)
    close_logger()
