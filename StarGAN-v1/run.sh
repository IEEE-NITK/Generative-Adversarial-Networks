while true
do
	source activate tensorflow_p36
	python main.py -c stargan_config.json
	source deactivate
done
