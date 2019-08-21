#!/usr/bin/env python

import numpy as np
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from Bio import SeqIO
import sys, re, h5py
print('done loading')

class hot_dna:
 def __init__(self,fasta):
   
  #check for and grab sequence name
  if re.search(">",fasta):
   name = re.split("\n",fasta)[0]
   sequence = re.split("\n",fasta)[1]
  else :
   name = 'unknown_sequence'
   sequence = fasta
  
  #get sequence into an array
  seq_array = array(list(sequence))
    
  #integer encode the sequence
  label_encoder = LabelEncoder()
  integer_encoded_seq = label_encoder.fit_transform(seq_array)
    
  #one hot the sequence
  onehot_encoder = OneHotEncoder(sparse=False)
  #reshape because that's what OneHotEncoder likes
  integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
  onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
  
  #add the attributes to self 
  self.name = name
  self.sequence = fasta
  self.integer = integer_encoded_seq
  self.onehot = onehot_encoded_seq
  self.classes = label_encoder.classes_

def main():
	if len(sys.argv) <= 1:
		encode(sys.stdin)
	else:
        	filename = sys.argv[1]
		encode(filename)

def encode(filename):
	depth = len(list(SeqIO.parse(filename, "fasta")))
	bp = len(list(SeqIO.parse(filename, "fasta"))[0].seq)
	l = np.ndarray((depth, 4, bp))
	index = 0

	for seq_record in SeqIO.parse(filename, "fasta"):
		output = hot_dna(str(seq_record.seq))
		output = output.onehot.transpose()

		if output.shape[0] != 4:
			missing = np.zeros((1, bp))
			d = {}
                        d.update(dict.fromkeys(['A', 'C', 'G', 'T'], missing))

			output = hot_dna(str(seq_record.seq))
			here = output.classes
			output = output.onehot.transpose()

			for letter, seq in zip(here, range(len(here))):
				d[letter] = output[seq, :]
							
			output = np.vstack((d['A'], d['C'], d['G'], d['T']))

		#switch	order from ACGT	to AGCT
                temp = np.copy(output[1, :])
                output[1, :] = output[2, :]
                output[2, :] = temp

		output = output.astype('float64')
		l[index] = output
		index+=1
		if index % 10000 == 0:
			print(index)

	prepath = filename.split('/')[0:-1]
	path = "/".join(prepath)
	presuffix = filename.split('/')[-1]
	suffix = path + '/' + presuffix.split('.')[0] + 'AGCT'

	#different for masked
	a = np.full((depth, 4, 427), 0.25, dtype = 'float64')
	b = np.full((depth, 4, 428), 0.25, dtype = 'float64')
	l = np.concatenate((a, l, b), axis = 2)

	np.save(suffix + '.npy', l)
	h5f = h5py.File(suffix + '.hdf5', 'w')
	h5f.create_dataset('testxdata', data = l) 	

if __name__ == '__main__':
   main()
