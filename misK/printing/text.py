import sys
import traceback

import sty

from misK.utils import BColors
import misK


def vprint(*args, end='\n'):
    """
        Works exactly as built-in print, except that it prints always on the same line as the previous one.
        Use a regular print to change line.
    """
    print(*args, end=end)
    sys.stdout.write("\033[F\033[K")


def verror(log=lambda *args, **kw: ''):
    """
        Takes care of the current error by printing the stack trace in red. Should be called when an error gets caught.
    """
    print(f"\n{BColors.CRED}{traceback.format_exc()}{BColors.ENDC}")
    log(traceback.format_exc())


def give_heading(text='', ll=6):
    """
        Gives a heading for pretty functional display.

        Args:
            text (str): a piece of text to include inside the heading. if text is longer than 'll', it will be cropped.
            ll (int): the width of the heading string.

        Return:
            (str) a string, i.e. the heading.
    """
    misK.utils.__LINE_NUMBER += 1
    return (sty.bg(100, 100, 100) if misK.utils.__LINE_NUMBER % 2 else sty.bg(200, 200, 200)) + ' ' + \
           sty.fg(0, 0, 0) + ("{: ^" + str(ll) + "}").format(text[:ll]) + ' ' + sty.fg.rs + sty.bg.rs + ' '


def strad(string):
    """
        Converts a string into its representation, i.e. integer, float or bool, if possible.

        Args:
            string (str): the string representation one wants to convert to its 'true' value.

        Return:
            (str, int, float, bool) the 'true' value of the input string representation.
    """
    if string.__class__.__name__ != "str":
        return string

    if string in ["True", "False"]:
        return True if string == "True" else False

    if string.isdigit():
        return int(string)

    elif string.replace('.', '', 1).isdigit():
        return float(string)

    return string
