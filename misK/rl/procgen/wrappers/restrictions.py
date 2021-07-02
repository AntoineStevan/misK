from misK.rl.procgen.wrappers.base import VecEnvWrapper


class LimitEpisode(VecEnvWrapper):
    def __init__(self, venv, max_steps, log=print):
        """
            Constructs a LimitEpisode instance, to allow the limiting of an agent in the environment.

            Args
            ----
            venv : ToBaselinesVecEnv or child
                the vectorized environment that needs some recording.
            max_steps : int
                the maximum number of steps allowed per episode before force quit.
            log : function, optional
                the log function to print strings. Defaults to built-in print.

            Returns
            -------
            self : LimitEpisode
                the constructed LimitEpisode object instance.
        """
        log(f"-> {self.__class__.__name__}")
        super().__init__(venv=venv)
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
            (obs, rewards, dones, infos) : (Tensor, float or int, bool, dict or None)
                gives the gym/procgen results for a step method.
        """
        obs, rewards, dones, infos = self.venv.step_wait()

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.current_step = 0
            dones = [True] * self.num_envs

        return obs, rewards, dones, infos

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
