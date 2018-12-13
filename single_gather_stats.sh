python gather_statistics.py --env $3 --samples $2 --nearly_silent True --no_visuals True --agent_to_eval "$1" --agent_type our_mlp |grep '^\[MAGIC NUMBER 87623123]' | tee -a "all_stats.log"
