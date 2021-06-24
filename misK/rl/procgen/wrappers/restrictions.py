from misK.rl.procgen.wrappers.base import VecEnvWrapper


class LimitEpisode(VecEnvWrapper):
    def __init__(self, env, max_steps, verbose=False):
        """
            Constructs a LimitEpisode instance, to allow the limiting of an agent in the environment.

            Args
            ----
                env : ToBaselinesVecEnv or child
                    the vectorized environment that needs some recording.
                max_steps : int
                    the maximum number of steps allowed per episode before force quit.
                verbose : bool, optional
                    triggers the verbose mode.

            Returns
            -------
            self : LimitEpisode
                the constructed LimitEpisode object instance.
        """
        if verbose:
            print(f"-> {self.__class__.__name__}", end=' ')
        super().__init__(venv=env)
        self.max_steps = max_steps
        self.current_step = 0

    def step_wait(self):
        """
            Wrapper for the step_wait method.
            Also increments the current episode step and check it against the maximum steps per episode.

            Args
            ----

            Returns
            -------
            (obs, reward, done, info) : (Tensor, float or int, bool, dict or None)
                gives the gym/procgen results for a step method.
        """
        obs, reward, done, info = self.venv.step_wait()

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.current_step = 0
            done = [True]

        return obs, reward, done, info

    def reset(self):
        """
            Wrapper for the reset method.
            Also resets the current episode step.

            Args
            ----

            Returns
            -------
            obs : (Tensor)
                gives the gym/procgen first observation in the environment.
        """
        self.current_step = 0
        return self.venv.reset()