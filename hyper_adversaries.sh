dir='weekend'
log='weekend_log'
itts=6
tests=1

parallel --delay 60 --load 50% --noswap --memfree 1G --jobs 10 --results logs/ --header : python rl_baseline.py weekend_test_shaped_winloss__cuttof{cutoff}__opp_mag{opp_mag}__me_mag{me_mag} --out-dir $dir/batch_1 --total-timesteps $itts --network mlp --reward me_pos~2~-100~{cutoff} --reward opp_mag~inf~{opp_mag}~smooth --reward me_mag~inf~-{me_mag}~smooth ::: cutoff 1 2 ::: opp_mag 0 0.1 1 2 10 ::: me_mag 0 0.1 1 2 10

bash gather_stats_all.sh $dir/batch_1/ $tests > $log.log



