dir='weekend'
log='weekend_log'
itts=3000000
tests=1000

parallel --delay 10 --load 100% --noswap --results logs/ --header : python rl_baseline.py weekend_test_shaped_winloss__cuttof{cutoff}__opp_mag{opp_mag}__me_mag{me_mag} --out-dir $dir/batch_1 --total-timesteps $itts --network mlp --reward me_pos~2~-100~{cutoff} --reward opp_mag~inf~{opp_mag}~smooth --reward me_mag~inf~-{me_mag}~smooth ::: cutoff 1 2 ::: opp_mag 0 0.1 1 2 10 ::: me_mag 0 0.1 1 2 10

bash gather_stats_all.sh $dir/batch_1/ $tests > $log.log



