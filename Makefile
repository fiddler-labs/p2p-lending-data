default:
	make install_requirements download_data process_data

install_requirements:
	pip3 install -r requirements.txt

download_data:
	kaggle datasets download --unzip -p raw_data -d wordsforthewise/lending-club/ -f accepted_2007_to_2018Q3.csv.gz

process_data:
	python3 create_p2p_dataset.py
