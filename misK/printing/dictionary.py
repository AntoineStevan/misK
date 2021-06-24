import os
import re

regex = re.compile(r"""
    \x1b     # literal ESC
    \[       # literal [
    [;\d]*   # zero or more digits or semicolons
    [A-Za-z] # a letter
    """, re.VERBOSE)


def hpprint(dicti, heading='', end='\n'):
    """
        Prints a dictionary in a two rows table, with each column representing a field of the dictionary.

        Args:
            dicti (dict): the actual dictionary to display.
            heading (str): a simple heading to make printing prettier.
            end (str): acts like the 'end' argument of built-in print function.

        Return:
            (None)
    """
    columns, _ = os.get_terminal_size()
    columns = 1024 if columns == 0 else columns

    regex.findall(heading)
    heading_l = len(str(regex.sub("", heading)))

    dict_repr = ""
    names, values, inter = '', '', ''
    for key, value in dicti.items():
        col_width = max(len(str(key)), len(str(value)))
        if heading_l + len(names) + col_width + 2 > columns:
            dict_repr += heading + names + '\n' + heading + values + '\n' + heading + inter[:-1] + '\n'
            names, values, inter = '', '', ''
        names += ("{: >" + str(col_width) + "} | ").format(key)
        if value.__class__.__name__ == "dict":
            value = list(value.keys())
        values += ("{: >" + str(col_width) + "} | ").format(str(value))
        inter += '-' * (col_width + 3)
    dict_repr += heading + names + '\n' + heading + values + '\n' + heading + inter[:-1]
    print(dict_repr, end=end)
