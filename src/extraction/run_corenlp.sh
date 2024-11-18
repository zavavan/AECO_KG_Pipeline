corenlp_process_pid=$!
cd ..
echo $corenlp_process_pid
python corenlp_extractor.py
kill -9 $corenlp_process_pid

