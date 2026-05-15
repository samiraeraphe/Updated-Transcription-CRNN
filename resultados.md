# Resultados da alteração de Greedy para Beam Search

Beam Width	WER (↓)	CER (↓)	BLEU Score (↑)
3	0.1473	0.0318	0.6288
4	0.1280	0.0271	0.6499
5	0.1208	0.0237	0.6801
6	0.1259	0.0286	0.6497

# Resultados da remoção do int_to_char[90] = ' '

Impressões:
Formato da saída do modelo: (5, 100, 91)  
Quantidade de caracteres no int_to_char: 90  
Maior índice no int_to_char: 89  

Greedy = true  
Média do WER: 0.0323513160538477  
Média do CER: 0.005323021762259378  
BLEU Score médio para o conjunto de teste: 0.9025  

Greedy false & beam_width = 5  
Média do WER: 0.16351466747036367
Média do CER: 0.033822548273738276
BLEU Score médio para o conjunto de teste: 0.6150
