from misK.rl.procgen.wrappers.base import VecEnvWrapper


class TrackAgent(VecEnvWrapper):
    def __init__(self, venv, log=print):
        """
            Constructs a TrackAgent instance, to allow the tracking of an agent.

            Args
            ----
            venv : ToBaselinesVecEnv or child
                the vectorized environment that needs some recording.
            log : function, optional
                the log function to print strings. Defaults to built-in print.

            Returns
            -------
            self : TrackAgent
                the constructed TrackAgent object instance.
        """
        log(f"-> {self.__class__.__name__}")
        super().__init__(venv=venv)
        self.agent = None

    def step_wait(self):
        """
            Wrapper for the step_wait method.

            Args
            ----

            Returns
            -------
            (obs, rewards, dones, infos) : (Tensor, float or int, bool, dict or None)
                gives the gym/procgen results for a step method.
        """
        obs, rewards, dones, infos = self.venv.step_wait()

        return obs, rewards, dones, infos

    def reset(self):
        """
            Wrapper for the reset method.

            Args
            ----

            Returns
            -------
            obs : (Tensor)
                gives the gym/procgen first observation in the environment.
        """
        return self.venv.reset()

    def track(self, agent):
        """
            Starts the agent tracking.

            Args
            ----
            agent : nn.Module or child
                the agent to track.

            Returns
            -------
            None
        """
        self.agent = agent
