import argparse
import difflib

from misK.printing.text import strad


class StoreDictKeyPair(argparse.Action):
    """
        A wrapper of the Action class from the argparse module. Allows to store argument in dictionaries.
    """

    def __init__(self, option_strings, dest, nargs='*', **kwargs):
        """
            A constructor for the StoreDictKeyPair class.

            Args:
                option_strings (list of strings): given by the parser -> all the possible flag names.
                dest (str): given by the parser -> the name of the destination of the dictionary.
                nargs (str): given by the parser -> the string representing the number of arguments possible.
                kwargs (dict[str, any]): given by the parser -> all the other arguments used by the parser
                    (choices, help, ...)

            Return:
                (StoreDictKeyPair) the constructed object.
        """
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

        choices_were_given = self.choices is not None and len(list(self.choices)) > 0

        # build the default dictionary using strings in choices, of the form "name:type:default".
        self._default = {}
        self._types = []
        if choices_were_given:
            for choice in self.choices:
                name, typ, default = choice.split(':')  # split the strings.
                self._types.append(typ)  # store the type.
                self._default[name] = default  # make a new name-default pair.

        self._nb_propositions = 2  # the number of choices printed by the auto completion on typos.

        # build the format.
        if choices_were_given:
            cols_w = []
            for i in range(len(list(self.choices))):
                col = max(len(list(self._default.keys())[i]), len(list(self._default.keys())[i]),
                          len(list(self._default.values())[i]), len(self._types[i]))
                cols_w.append(col)
            cols_f = ["{: ^" + str(col) + '}' for col in cols_w]

            line1 = "command | python src/main.py --{} ".format(self.dest) + ' '.join(["{}="] * len(list(self.choices)))
            line2 = " names  | " + ' | '.join(["{}"] * len(list(self.choices)))
            line3 = "default | " + ' | '.join(["{}"] * len(list(self.choices)))
            line4 = " types  | " + ' | '.join(["{}"] * len(list(self.choices)))

            lines = [line1.format(*self._default.keys()), "{}"] + [line.format(*cols_f) for line in
                                                                   [line2, line3, line4]]
            for i, content in enumerate([[], ["{}"], self._default.keys(), self._default.values(), self._types]):
                lines[i] = lines[i].format(*list(content))

            lines[1] = '+'.join(['-' * 8] + ['-' * (col_w + 2) for col_w in cols_w])
            self.format = '\n'.join(lines)
        else:
            self.format = f"python path/to/main.py [*] KEY1=VAL1 KEY2=VAL2 ... KEYN=VALN "

        self.choices = None

    def __call__(self, parser, namespace, values, option_string=None):
        """
            Called when the parser parses the arguments.

            Args:
                parser (ArgumentParser): the actual parser.
                namespace (Namespace): the namespace of the parser, used to attribute the built dictionary to
                    destination.
                values (list of str): all the values given by the user.
                option_string (str): the user chosen option string.

            Return:
                (None)

            Throws:
                (ValueError) raised when input is not of the form KEY=VAL, treated as a Warning.
                (TypeError) raised when a user chosen key is not available.
        """
        my_dict = dict(self._default)
        k = ''
        for kv in values:
            try:
                k, v = kv.split("=")
                if len(self._default) > 0 and k not in self._default.keys():
                    raise TypeError()
                if v != '':
                    my_dict[k] = v
            except ValueError:
                warning_msg = f"usage of {' or '.join(self.option_strings)}:" + '\n' + self.format
                raise Warning("CUSTOM" + warning_msg)
            except TypeError:
                error_msg = f"unknown argument name '{k}' for {' and '.join(self.option_strings)}\n"
                matches = difflib.get_close_matches(k, self._default.keys(), n=self._nb_propositions)
                if matches:
                    error_msg += f"\tdid you mean: {' or '.join(matches)}?"
                else:
                    error_msg += f"possible argument names for " + \
                                 "{}:\n\t{}".format(' and '.join(self.option_strings), ", ".join(self._default.keys()))
                raise ValueError("CUSTOM" + error_msg)

        for k, v in my_dict.items():
            my_dict[k] = strad(v)

        setattr(namespace, self.dest, my_dict)
