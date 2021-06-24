import difflib

import yaml

from misK.printing.dictionary import hpprint
from misK.utils import BColors


def load_params(path, adjust=None, verbose=False):
    """
        Used to load parameters from a .yaml config file.

        Args
        ----
        path : str
            a path to the config file. does not need to end with '.yaml', but it can.
        adjust : dict
            an adjustment dictionary to make punctual and custom correction of the config parameters.
        verbose : bool
            triggers the verbose mode.

        Returns
        -------
        params : dict
            the parameters dictionary from 'path' and adjusted with 'adjust'.
    """
    # correct the path format.
    path = path if path.endswith(".yaml") else path + ".yaml"

    if verbose:
        print("loading parameters from", path, end='... ')
    # open and load the parameters.
    with open(path, 'r') as file:
        kwargs = yaml.full_load(file)
    if verbose:
        print("done")

    # apply the given adjustments if needed.
    if adjust is not None and len(adjust) > 0:
        if verbose:
            print(f"adjusts performed on parameters from {path}", end='')
        for key in adjust:
            if verbose:
                print(" -", key, end='')
            kwargs[key] = adjust[key]
        if verbose:
            print()

    if verbose:
        show_args(dict(
            kwargs=kwargs,
        ), prt_name=False)

    return kwargs


def show_args(args, color="CBLUE2", prt_name=True, end=''):
    # basic printing of the arguments parsed.
    try:
        print(BColors.__dict__[color], end='')
    except KeyError as ke:
        colors = [col for col in list(BColors.__dict__.keys()) if "__" not in col]
        closest_colors = difflib.get_close_matches(color, colors, n=3)
        print(BColors.CRED + f"{ke} is not a valid color. Dic you mean {', '.join(closest_colors)}?" + BColors.ENDC)
    finally:
        for name, arg in args.items():
            if arg.__class__.__name__ in ["int", "float", "bool"]:
                if prt_name:
                    print(name)
                print(arg)
            elif len(arg) > 0:
                if prt_name:
                    print(name)
                hpprint(arg)
        print(BColors.ENDC, end=end)
