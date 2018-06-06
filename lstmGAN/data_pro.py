#!/bin/python
# encoding: utf-8
import config
import numpy as np
import time
import math
import os
import torch
from torch.autograd import Variable
import random
import pdb
import sys
import cPickle as pickle
from collections import Counter
import cPickle as pickle
import pdb
import time

class dataSet(object):
	def __init__(self):
		self.en_words_count = Counter()
		self.ch_words_count = Counter()
		#self.relations = []
		self.other_label = "none"
		#self.relation.append(self.other)
		self.en_max_len = config.en_max_len
		self.ch_max_len = config.ch_max_len
		self.batch = config.BATCH

	def load_pkl(self):
		pklf = open(config.data_pkl, 'rb')
		self.train_vec = pickle.load(pklf)
		#self.EN_chtrain_vec = pickle.load(pklf)
		self.CH_train_vec = pickle.load(pklf)
		self.CH_test_vec = pickle.load(pklf)
		self.relations = pickle.load(pklf)
		self.EN_size = pickle.load(pklf)
		self.CH_size = pickle.load(pklf)
		config.tag_size = pickle.load(pklf)
		labels = pickle.load(pklf)
		maxlen = pickle.load(pklf)
		pklf.close()
		pklf = open(config.emb_pkl, 'rb')
		config.EN_embeddings = pickle.load(pklf)
		config.CH_embeddings = pickle.load(pklf)
		pklf.close()
		self.train_labels, self.CH_test_labels, self.CH_train_labels = labels
		config.EN_maxlen, config.CH_maxlen = maxlen 
		config.EN_vocab_size, config.NER_size = self.EN_size
		config.CH_vocab_size = self.CH_size
		config.EN_dist_size = 124
		config.CH_dist_size = 124
		#self.train_vec = self.merge(EN_train_vec, EN_chtrain_vec)
			

	def preprocess(self):
		start = time.time()
		self.relations = self._load_file(config.relation_path)
		print >> sys.stderr, "loading data..."
		EN_train_data = self.load_data(config.EN_train_file)
		EN_chtrain_data = self.load_data(config.EN_chtrain_file)
		CH_test_data = self.load_data(config.CH_test_file)
		CH_train_data = self.load_data(config.CH_train_file)
		#pdb.set_trace()
		self.en_type2id(EN_train_data[4]+CH_test_data[4]+CH_train_data[4])
		self.buildfea2id(EN_train_data, [], [], True)
		self.buildfea2id(EN_chtrain_data, CH_test_data, CH_train_data, False)
		print >> sys.stderr, "loading embeddings..."
		now = time.time()
		config.EN_embeddings, self.EN_words2id = self.load_embedding(config.EN_vocab_file, config.EN_emb_file, True)
		config.CH_embeddings, self.CH_words2id = self.load_embedding(config.CH_vocab_file, config.CH_emb_file, False)
		#pdb.set_trace()
		EN_train_vec, EN_train_labels = self.vector(EN_train_data, True)
		EN_chtrain_vec, EN_chtrain_labels = self.vector(EN_chtrain_data, False)
		self.CH_train_vec, self.CH_train_labels = self.vector(CH_train_data, False)
		self.train_labels = EN_train_labels
		self.train_vec = self.merge(EN_train_vec, EN_chtrain_vec)
		self.CH_test_vec, self.CH_test_labels = self.vector(CH_test_data, False)
		
		config.tag_size = len(self.relations)
		self.EN_size = (config.EN_vocab_size, config.NER_size)
		self.CH_size = config.CH_vocab_size
		labels = self.train_labels, self.CH_test_labels, self.CH_train_labels
		maxlen = config.EN_maxlen, config.CH_maxlen
		print >> sys.stderr, "dump data..."
		'''
		pklf = open(config.data_pkl, 'wb')
		pickle.dump(self.train_vec, pklf, 2)
		#pickle.dump(self.EN_chtrain_vec, pklf, -1)
		pickle.dump(self.CH_train_vec, pklf, -1)
		pickle.dump(self.CH_test_vec, pklf, -1)
		pickle.dump(self.relations, pklf, -1)
		pickle.dump(self.EN_size, pklf, -1)
		pickle.dump(self.CH_size, pklf, -1)
		pickle.dump(config.tag_size, pklf, -1)
		pickle.dump(labels, pklf, -1)
		pickle.dump(maxlen, pklf, -1)

		pklf.close()
		pklf = open(config.emb_pkl, 'wb')
		pickle.dump(config.EN_embeddings, pklf, 2)
		pickle.dump(config.CH_embeddings, pklf, -1)
		pklf.close()
		'''
		print >> sys.stderr, "loading time {}".format(self.timeSince(start))
	def merge(self, data1, data2):
		train_vec = []
		for vec1, vec2 in zip(data1, data2):
			vector = []
			vector.extend(vec1)
			vector.extend(vec2)
			train_vec.append(vector)
		return train_vec
	def _load_file(self, file):
		relations = []
		with open(file, 'r') as f:
			for line in f:
				line = line.strip()
				relations.append(line)
		f.close()
		return relations


	def load_embedding(self, vocab_file, emb_npy, en=True):
		vocab = []
		if en:
			vocab = self.EN_vocab
		else:
			vocab = self.CH_vocab
		emb_words = self._load_file(vocab_file)
		embeddings = np.load(emb_npy)
		pre_emb2id = {}
		for id, w in enumerate(emb_words):
			pre_emb2id[w] = id

		words2id = {}
		word_embed = []
		word_embed.append(np.zeros(config.words_dim))
		for idx,w in enumerate(vocab):
			if w in emb_words:
				vec = embeddings[pre_emb2id[w]]
				word_embed.append(vec)
			else:
				vec = np.random.normal(0,0.1,config.words_dim)
				word_embed.append(vec)
			words2id[w] = idx+1
		word_embed = np.array(word_embed,  dtype=float)
		
		return word_embed, words2id

	def load_data(self, file):
		e1_pos = []
		e2_pos = []
		en_type = []
		#e2_type = []
		_sentences = []
		_pos = []
		_en = []
		labels = []
		fi = open(file, 'r')
		for line in fi:
			line = line.strip()
			line = line.split("\t")
			labels.append(self.relations.index(line[0]))
			_sentences.append(line[9].split(" "))
			try:
				e1_pos.append((int(line[5]), int(line[6])))
				e2_pos.append((int(line[7]), int(line[8])))
			except:
				pdb.set_trace()
			en_type.append((line[3], line[4]))
		return labels, _sentences, e1_pos, e2_pos, en_type
	def en_type2id(self, types):
		ner_types = set()
		ner2id = {}
		for ner1, ner2 in types:
			ner_types.add(ner1)
			ner_types.add(ner2)
		ner_types = sorted(list(ner_types))
		for idx, ner in enumerate(ner_types):
			ner2id[ner] = idx
		config.NER_size = len(ner_types)
		self.ner2id = ner2id
		return
	def buildfea2id(self, data1, data2, data3, EN=True):
		data = []

		maxlen = 0
		_sentences = data1[1]
		if not EN:
			_sentences = data1[1]+data2[1]+data3[1]
			
		vocab =set()
		
		for sent in _sentences:
			if maxlen < len(sent):
				maxlen = len(sent)
			for w in sent:
				w = w.lower()
				vocab.add(w)
		vocab = sorted(list(vocab))

		if EN:
			self.EN_vocab = vocab
			config.EN_maxlen = maxlen
			config.EN_vocab_size = len(vocab)+1
			
		else:
			self.CH_vocab = vocab
			config.CH_maxlen = maxlen
			config.CH_vocab_size = len(vocab)+1
		return

	def vector(self, data, en=True):
		def distance(dist1, dist2):
			'''convert relative distance to positive number
			-60), [-60, 60], (60
			 '''
			# FIXME: FLAGS.pos_num
			if dist1 < -60:
				return 1
			elif dist1 >= -60 and dist1 <= 60:
				return dist1 + 62   
			return 123
		config.EN_dist_size = 124
		config.CH_dist_size = 124
		words2id = {}
		ner2id = self.ner2id
		maxlen = 0
		if en:
			words2id = self.EN_words2id
			maxlen = config.EN_maxlen
		else:
			words2id = self.CH_words2id
			maxlen = config.CH_maxlen
		_sentences = data[1]
		labels = data[0]
		_e1_pos = data[2]
		_e2_pos = data[3]
		ner_types = data[4]
		data_vec = []
		for idx, (sent, p1, p2, (ner1, ner2)) in enumerate(zip(_sentences, _e1_pos, _e2_pos, ner_types)):
			sent_vec = [0]*maxlen
			dist1_vec = [0]*maxlen
			dist2_vec = [0]*maxlen
			ner_vec = []
			vec = [words2id[w.lower()] for w in sent]
			dist1 = [distance(i-p1[0], i-p1[1]) for i in range(len(sent))]
			dist2 = [distance(i-p2[0], i-p2[1]) for i in range(len(sent))]
			sent_vec[:len(sent)] = vec
			dist1_vec[:len(sent)] = dist1
			dist2_vec[:len(sent)] = dist2
			ner_vec.append(ner2id[ner1])
			ner_vec.append(ner2id[ner2])
			vector = []
			#pdb.set_trace()
			vector.extend(sent_vec)
			vector.extend(dist1_vec)
			vector.extend(dist2_vec)
			vector.extend(ner_vec)
			
			data_vec.append(vector)
		#pdb.set_trace()
		data_vec = np.array(data_vec, dtype=int)
		labels = np.array(labels, dtype=int)
		return data_vec, labels

	def getLabel(self, categories_list):
		#pdb.set_trace()
		return [self.relations[x] for x in categories_list]

	def timeSince(self, since):
		now = time.time()
		s = now-since
		m = math.floor( s / 60)
		s -= m * 60
		return '%dm %ds' % (m, s)
	def Score(self, ans_AT, gold_AT, flag=False):
		#labels = self.relations
		'''
		for relation in self.relations:
			if "(" in relation:
				relation = relation[:relation.index("(")]
			if relation not in labels:
				labels.append(relation)
		'''
		#pdb.set_trace()
		relation_sum = len(self.relations)-1
		#sys.stderr.write("categories sum is %d\n" %(relation_sum))
		row = 0
		AvgF = 0.0
		AvgP = 0.0
		AvgR = 0.0
		TP_ = 0
		FP_ = 0
		FN_ = 0
		
		for relation in self.relations:
			row = relation
			relation_count = 0
			out_count = 0
			TP = 0
			TN = 0
			FP = 0
			FN = 0 
			P = 0.0
			R = 0.0
			F = 0.0
			for i in range(len(gold_AT)):
				if relation == ans_AT[i]:
					out_count += 1

				if relation == gold_AT[i]:

					relation_count += 1
					if ans_AT[i] == gold_AT[i]:
						TP += 1
					else:
						FN += 1
				
			FP = out_count - TP
			if (TP + FP) != 0:
				P = 1.0 * TP / (TP + FP)
			if (TP + FN) != 0:
				R = 1.0 * TP / (TP + FN)
			if (P + R) != 0:
				F = 2.0 * P * R / (P + R)
			if relation != self.other_label:
				AvgP += P
				AvgR += R
				TP_ += TP
				FP_ += FP
				FN_ += FN
			if flag:
				sys.stderr.write( "##%-12s samples:%-6s\tP=%.4f(%4d/%4d)\tR=%.4f(%4d/%4d)\tF=%.4f\t%4d,%4d%4d\n" % (
					relation, relation_count, P, TP, (TP+FP), R, TP, (TP+FN), F, TP, FP, FN))
		if (TP_+FP_) == 0: TP_ = 0.000001
		if (TP_+FN_) == 0: FN_ = 1.0
		P = 1.0*TP_/(TP_+FP_)
		R = 1.0*TP_/(TP_+FN_)
		
		MicroF = 2.0*P*R/(P+R)
		sys.stderr.write ("\nMicroPRF:\t%.4f (%4d/%4d), %.4f (%4d/%4d), %.4f\n" % (P, TP_, TP_+FP_, R, TP_, TP_+FN_, MicroF))
		#pdb.set_trace()
		AvgP /= relation_sum
		AvgR /= relation_sum
		if (AvgP + AvgR) == 0: AvgP = 0.00001
		MacroF = 2.0 * AvgP * AvgR / (AvgP + AvgR)
		sys.stderr.write ("MacroPRF:\t%.4f, %.4f, %.4f\n" % (AvgP, AvgR, MacroF))
		return AvgP, AvgR, MacroF
















