def dataproc():
	with open('F:/Files/Program/py/TensorFlow/Model2/ori.txt', 'r') as f:
		str= f.read()
		with open('F:/Files/Program/py/TensorFlow/Model2/ori_proc.txt', 'w') as fr:
			flag=1
			for char in str:
				if char ==' ' and flag ==1:
					fr.write('\n')
					flag= 0
				elif char =='\n':
					fr.write('')
				else:
					fr.write(char)
					if char == '.':
						flag= 1

	with open('F:/Files/Program/py/TensorFlow/Model2/detor.txt', 'r') as f:
		str= f.read()
		with open('F:/Files/Program/py/TensorFlow/Model2/detor_proc.txt', 'w') as fr:
			flag=1
			for char in str:
				if char ==' ' and flag ==1:
					fr.write('\n')
					flag= 0
				elif char =='\n':
					fr.write('')
				else:
					fr.write(char)
					if char == '.':
						flag= 1

	with open('F:/Files/Program/py/TensorFlow/Model2/calc.txt', 'r') as f:
		str= f.read()
		with open('F:/Files/Program/py/TensorFlow/Model2/calc_proc.txt', 'w') as fr:
			flag=1
			for char in str:
				if char ==' ' and flag ==1:
					fr.write('\n')
					flag= 0
				elif char =='\n':
					fr.write('')
				else:
					fr.write(char)
					if char == '.':
						flag= 1


