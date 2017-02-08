#phrase_add
python phrase_add.py ../embedding/vector.sg50.100k.eng.w 50

#phrase_lstm_mse.py / phrase_lstm_mm.py / phrase_matrix_mse.py / phrase_matrix_mm.py
python phrase_lstm_mse.py -batchsize 100 -lamda_ww 0.0 -lamda_w 1.0 -wordfile ../embedding/vector.sg50.100k.eng -dim 50
