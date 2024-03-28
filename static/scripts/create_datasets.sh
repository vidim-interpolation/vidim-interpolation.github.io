wget https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip
unzip DAVIS-data.zip
python3 generate_davis-7.py --davis_source_path DAVIS --davis7_target_path DAVIS-7
python3 generate_ucf101-7.py --ucf101_7_target_path UCF101-7
