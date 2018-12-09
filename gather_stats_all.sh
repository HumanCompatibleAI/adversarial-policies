
for file in $1/*
do
	  python gather_statistics.py --samples $2 --nearly_silent True --no_visuals True --agent_to_eval "$file" --agent_type our_mlp |grep '^\[MAGIC NUMBER 87623123]' | tee -a "all_stats.log" 
done
