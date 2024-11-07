corenlp_process_pid=$!
cd ..
echo $corenlp_process_pid
python corenlp_extractor.py 4
kill -9 $corenlp_process_pid

