
from utils import load_agent, CurryEnv, DelayedLoadEnv
from utils import get_env_and_policy_type, make_session, get_trained_kicker_locations

from gather_statistics import get_emperical_score, get_agent_any_type



samples = 20

env, pol_type = get_env_and_policy_type('kick-and-defend')


sess = make_session()
with sess:
    known_agent = load_agent('agent-zoo/kick-and-defend/defender/agent2_parameters-v1.pkl', pol_type,
                             "known_policy", env, 1)

    attacked_agent = load_agent(get_trained_kicker_locations()[1], pol_type, "attacked", env, 0)

    #TODO Load Agent should be changed to "load_zoo_agent"


    #TODO Below is test for delayed start

    newenv = DelayedLoadEnv(env, get_trained_kicker_locations()[1], pol_type, "attacked3", 0, sess)
    #newenv = HackyFixForGoalie(newenv)

    trained_agent = get_agent_any_type('our_mlp', 'rando-ralph', pol_type, env)

    ties, win_loss = get_emperical_score(newenv, [trained_agent], samples, render=True, silent=True)
    #TODO Above is test for delayed start





    agents = [attacked_agent, known_agent]
    ties, win_loss = get_emperical_score(env, agents, samples, render=True, silent=True)

    print("[MAGIC NUMBER 87623123] In {} trials {} acheived {} Ties and winrates {}".format(samples, 'known agent got',
                                                                                            ties, win_loss))

    #trained_agent = get_agent_any_type('our_mlp', 'rando-ralph', pol_type, env)

    #agents = [attacked_agent, trained_agent]
    #ties, win_loss = get_emperical_score(env, agents, samples, render=True, silent=True)

    print("[MAGIC NUMBER 87623123] In {} trials {} acheived {} Ties and winrates {}".format(samples, 'rando ralph got',
                                                                                            ties, win_loss))
