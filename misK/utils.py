import sty

from misK.printing.dictionary import hpprint

__LINE_NUMBER = 0


def print_format_table():
    """
        prints table of formatted text format options
    """
    for style in range(8):
        for fg in range(30, 38):
            s1 = ''
            for bg in range(40, 48):
                format = ';'.join([str(style), str(fg), str(bg)])
                s1 += '\x1b[%sm %s \x1b[0m' % (format, format)
            print(s1)
        print('\n')


class BColors:
    """ A few possible colors in the terminal. """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    CEND = '\33[0m'
    CBOLD = '\33[1m'
    CITALIC = '\33[3m'
    CURL = '\33[4m'
    CBLINK = '\33[5m'
    CBLINK2 = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK = '\33[30m'
    CRED = '\33[31m'
    CGREEN = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE = '\33[36m'
    CWHITE = '\33[37m'

    CBLACKBG = '\33[40m'
    CREDBG = '\33[41m'
    CGREENBG = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG = '\33[46m'
    CWHITEBG = '\33[47m'

    CGREY = '\33[90m'
    CRED2 = '\33[91m'
    CGREEN2 = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2 = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2 = '\33[96m'
    CWHITE2 = '\33[97m'

    CGREYBG = '\33[100m'
    CREDBG2 = '\33[101m'
    CGREENBG2 = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2 = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2 = '\33[106m'


# +== EXPERIMENTS ======================================================================================================
def ppprint(text, fg=(), bg=(), style='', end='\n'):
    if fg is not ():
        text = sty.fg(*fg) + text + sty.fg.rs
    if bg is not ():
        text = sty.bg(*bg) + text + sty.bg.rs
    if style in sty.ef.__dict__:
        text = sty.ef.__dict__[style] + text
    print(text + sty.rs.rs, end=end)


def lprint(text: str):
    global __LINE_NUMBER
    # print((sty.ef.italic if __LINE_NUMBER % 2 else '') + str(text) + sty.ef.rs)
    __LINE_NUMBER += 1


def _sprint(msg, style, fg, bg):
    return "\x1b[%sm %s \x1b[0]m" % (';'.join([str(style), str(fg), str(bg)]), msg)


def sprint(msg, style, fg, bg):
    print(_sprint(msg, style, fg, bg))


# =====================================================================================================================+


if __name__ == "__main__":
    # print_format_table()
    # print()
    # sprint("coucou", 3, 30, 47)
    ppprint("red over blue", fg=(255, 0, 0), bg=(0, 0, 255))
    a = 0
    ppprint("======================================== test ========================================", fg=(a, a, a),
            bg=(255 - a, 255 - a, 255 - a, 200))
    ppprint("======================================== test ========================================", bg=(a, a, a, 128),
            fg=(255 - a, 255 - a, 255 - a))

    for i in range(10):
        lprint(i)

    d = {'distribution_mode': 'hard',
         'env_name': 'coinrun',
         'num_envs': 1,
         'num_levels': 1,
         'render_mode': 'rgb_array',
         'restrict_themes': True,
         'start_level': 0,
         'use_backgrounds': True,
         'use_monochrome_assets': False,
         'use_sequential_levels': True}

    hpprint(d)


def dum():
    """
        Doc

        Parameters
        ----------
        param: types
            doc

            .. versionadded: version

        Returns
        -------

        See also
        --------
        function_name : desc

        Notes
        -----
        blabla

        Examples
        --------
        blabla
    """
