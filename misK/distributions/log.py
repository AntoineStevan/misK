import os

from numpy import ceil as np_ceil

from torch.nn.functional import softmax as tnnf_softmax


class ProbabilityDistributionLogger:
    def __init__(self, actions, trials, save_probas=None, show_probas=False, column_width=10):
        """
            ProbabilityDistributionLogger constructor.

            Args:
            -----
            trials : int, optional
                the number of episodes of the evaluation.
            trials : int, optional
                the number of episodes of the evaluation.
            save_probas : bool, optional
                the filename where to save the probability distributions. If None or '', no saving will be performed.
            show_probas : bool, optional
                tells whether to print the probability distributions of the agent in the terminal.
            column_width : int, optional
                the width of the columns used in 'show_probas' mode.

            Returns
            -------
            self : ProbabilityDistributionLogger
                a new instance of the ProbabilityDistributionLogger class.
        """
        self.save_probas = save_probas
        self.show_probas = show_probas
        self.file, self.labels, self.terminal_format, self.file_format, self.col_widths = None, None, None, None, []
        labels = []

        if self.save_probas:
            self.file = open(self.save_probas, 'w')

        if self.save_probas or self.show_probas:
            self.file_col_widths = [max(4, len(str(500 * trials))), max(2, len(str(trials)))]
            self.file_col_widths += [column_width] * len(actions)
            self.file_format = '|'.join(["{: ^" + str(col_width) + "}" for col_width in self.file_col_widths])
            labels = ["step", "ep"] + actions
            current_width, terminal_width = len(self.file_format.format(*labels)), os.get_terminal_size()[0]
            if current_width > terminal_width:
                print(f"-- TOO WIDE FOR TERMINAL-- maximum allowed of {terminal_width}: got {current_width}")
                self.terminal_col_widths = [max(4, len(str(500 * trials))), max(2, len(str(trials)))]
                new_col_width = (terminal_width - sum(self.terminal_col_widths)) // len(actions) - 2
                self.terminal_col_widths += [new_col_width] * len(actions)
                self.terminal_format = '|'.join(
                    ["{: ^" + str(col_width) + "}" for col_width in self.terminal_col_widths])
                print(f"\tgoing from {column_width} to {new_col_width}.")
            else:
                self.terminal_format = self.file_format[::]
                self.terminal_col_widths = self.file_col_widths[::]

        if self.show_probas:
            print(self.terminal_format.format(*labels))
            print(self.terminal_format.replace('|', '+').format(*([''] * len(self.terminal_col_widths))).replace(' ',
                                                                                                                 '-'))
        if self.save_probas:
            self.file.write(self.file_format.format(*labels) + '\n')
            self.file.write(
                self.file_format.replace('|', '+').format(*([''] * len(self.file_col_widths))).replace(' ', '-') + '\n')

    def log(self, agent, obs, frame, episode):
        """
            Logs the probability distribution given by the agent when given an observation 'obs' in episode 'episode'
            and at frame 'frame'.
            Also returns the probability distributions for further use.

            Args:
            -----
            agent : torch.nn.Module or child
                the policy, or equivalently the agent, under evaluation.
            obs : Tensor
                an input observation from the environment.
            frame : int
                the number of frames elapsed since beginning of training.
            episode : int
                the number of the current episode, starting from 1.

            Returns:
            --------
            probas : list of floats
                the probability distribution over the action space of the agent given the observation.
        """
        probas = None
        if self.show_probas or self.save_probas:
            probas = tnnf_softmax(agent.categorize(obs).logits, dim=1)[0].tolist()
            term_bars = [frame, episode] + ['.' * np_ceil(proba * col_width).astype(int) for proba, col_width in
                                            zip(probas, self.terminal_col_widths[2:])]
            file_bars = [frame, episode] + ['.' * np_ceil(proba * col_width).astype(int) for proba, col_width in
                                            zip(probas, self.file_col_widths[2:])]
            if self.show_probas:
                print(self.terminal_format.format(*term_bars))
            if self.save_probas:
                self.file.write(self.file_format.format(*file_bars) + '\n')

        return probas

    def close(self, agent):
        """
            Closes the logger, namely its file.

            Args:
            ----
            agent : torch.nn.Module or child
                the policy, or equivalently the agent, under evaluation.

            Returns:
            --------
            None
        """

        if self.save_probas:
            self.file.close()
            print(f"probability distributions of {agent.__class__.__name__} stored in {self.save_probas}")