from gym_compete.new_envs import SumoEnv


class SumoAutoContactEnv(SumoEnv):
    """
    Same as SumoEnv but agents automatically contact one another.
    This is so that falling or exiting the stage without touching
    the opponent counts as a loss and not a tie.
    """

    def reset(self, margins=None, version=None):
        ob = super(SumoAutoContactEnv, self).reset(margins, version)
        self.agent_contacts = True
        return ob
