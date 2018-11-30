AGENTS="pretrainedv1 mlp_train_default_shape mlp_train_no_shape out_lstm_rand out_random_const"
START_EP=3
END_EP=23

for agent in ${AGENTS}; do
	rm mylist.txt >/dev/null 2>&1; touch mylist.txt
	for i in $(seq -f "%06g" $START_EP $END_EP); do
		echo "file 'videos/${agent}/video.${i}.mp4'" >> mylist.txt
	done
	rm videos/${agent}.mp4
	ffmpeg -f concat -i mylist.txt -c copy videos/${agent}.mp4
done

rm mylist.txt
