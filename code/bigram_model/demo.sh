# phrase_add.py
python phrase_add.py ../embedding/vector.sg50.100k.eng 50

# phrase_matrix_mse.py / phrase_matrix_mm.py / phrase_recnn_mse.py / phrase_recnn_mm.py
python phrase_matrix_mse.py -batchsize 100 -lamda_ww 0.0 -lamda_w 1.0 -wordfile ../embedding/vector.sg50.100k.eng -dim 50
