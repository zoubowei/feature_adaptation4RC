#!/bin/python
# encoding: utf-8
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import config
import pdb
import torch.nn.init as init

class GAN(nn.Module):
	def __init__(self, tagset_size,  kernel_num, kernel_sizes1, kernel_sizes2):
		super(GAN, self).__init__()
		Ci = config.words_dim + 2 * config.dist_dim
		Co = kernel_num
		self.ENword_embeds = nn.Embedding(config.EN_vocab_size, config.words_dim, padding_idx=0)
		self.ENword_embeds.weight = nn.Parameter(torch.from_numpy(config.EN_embeddings.astype('float32')))
		self.ENdist_embeds = nn.Embedding(config.EN_dist_size, config.dist_dim, padding_idx=0)
		
		self.CHword_embeds = nn.Embedding(config.CH_vocab_size, config.words_dim, padding_idx=0)
		self.CHword_embeds.weight = nn.Parameter(torch.from_numpy(config.CH_embeddings.astype('float32')))
		self.CHdist_embeds = nn.Embedding(config.CH_dist_size, config.dist_dim, padding_idx=0)
		self.ner_embeds = nn.Embedding(config.NER_size, config.ner_dim)
		self.Drop = nn.Dropout(config.dropout)
		#if config.CNN:
		self.M1_conv = nn.ModuleList([nn.Conv1d(Ci, Co, K) for K in kernel_sizes1])
		self.M2_conv = nn.ModuleList([nn.Conv1d(Ci, Co, K) for K in kernel_sizes2])
		hidden_size1 = len(kernel_sizes1) * kernel_num 
		hidden_size2 = hidden_size1 +2* config.ner_dim
		self.hidden2tag = nn.Linear(hidden_size2, tagset_size) 
		self.hidden2diff = nn.Linear(hidden_size1, 2) 
		#self.M2_fc1 = nn.Linear()
		self.softmax = nn.Softmax()
		self.Drop = nn.Dropout(config.dropout)
		self.init_weight()

	def EMB(self, BATCH, EN= True):
		batch_sent, batch_dist1, batch_dist2, batch_ner = BATCH
		#word_embeds = self.ENword_embeds
		#if not en:
		word_embeds, dist_embeds = self.CHword_embeds, self.CHdist_embeds
		if EN:
			word_embeds, dist_embeds = self.ENword_embeds, self.ENdist_embeds
		ner_tensor = self.ner_embeds(batch_ner)
		batch_tensor = word_embeds(batch_sent)
		batch_dist1 = dist_embeds(batch_dist1)
		batch_dist2 = dist_embeds(batch_dist2)
		batch_tensor = torch.cat((batch_tensor, batch_dist1, batch_dist2), 2)
		if config.NER:
			ner_tensor = self.ner_embeds(batch_ner)
		#pdb.set_trace()
		
		return batch_tensor, ner_tensor
	def M1_CNN(self, EN_BATCH, train):
		batch_tensor, ner_tensor = self.EMB(EN_BATCH, True)
		ner_tensor = ner_tensor.view(-1, config.ner_dim * 2)
		if train:
			batch_tensor = self.Drop(batch_tensor)
		batch_tensor = torch.transpose(batch_tensor, 1, 2) # (B , Dim, Length)
		convx = []
		for id,conv in enumerate(self.M1_conv):
			convx.append(F.tanh(conv(batch_tensor)))   # (B, Co, Lout)*len(Ks)
		x = [torch.max(xi, 2)[0] for xi in convx] #(B, Co)*len(Ks)
		x = torch.cat(x,1)
		'''
		if train:
			x = self.Drop(x)
		'''
		#x = torch.cat((x, ner_tensor), -1)
		#hidden = F.tanh(self.M1_fc1(x))
		return x, ner_tensor

	def M2_CNN(self, CH_BATCH, train):
		batch_tensor, ner_tensor = self.EMB(CH_BATCH, False)
		ner_tensor = ner_tensor.view(-1, config.ner_dim * 2)
		if train:
			batch_tensor = self.Drop(batch_tensor)
		batch_tensor = torch.transpose(batch_tensor, 1, 2) # (B , Dim, Length)
		convx = []
		for id,conv in enumerate(self.M2_conv):
			convx.append(F.tanh(conv(batch_tensor)))   # (B, Co, Lout)*len(Ks)
		x = [torch.max(xi, 2)[0] for xi in convx] #(B, Co)*len(Ks)
		x = torch.cat(x,1)
		#x = torch.cat((x, ner_tensor), -1)
		'''
		if train:
			x = self.Drop(x)
		'''
		#x = F.tanh(self.M2_fc1(x))
		return x, ner_tensor
	def diff(self, hidden):
		#pdb.set_trace()
		logits = self.hidden2diff(hidden)
		tag_out = self.softmax(logits)
		score, labels = torch.max(tag_out, 1)
		return logits, labels, score

	def classifier(self, hidden):
		logits = self.hidden2tag(hidden)
		tag_out = self.softmax(logits)
		score, labels = torch.max(tag_out, 1)
		return logits, labels, score

	def forward(self, EN_BATCH, CH_BATCH, train = True):
		diff_log = []
		class_log = []
		class_labels = []
		CH_hidden = []
		if train:
			EN_hidden, ner_tensor = self.M1_CNN(EN_BATCH, train)
			CH_hidden, _ = self.M2_CNN(CH_BATCH, train)
			hidden = torch.cat((EN_hidden, CH_hidden), 0)
			#pdb.set_trace()
			diff_log, diff_labels, _ = self.diff(hidden)
			EN_hidden = torch.cat((EN_hidden, ner_tensor), -1)
			CH_hidden = torch.cat((CH_hidden, ner_tensor), -1)
			cat = torch.cat((EN_hidden, CH_hidden), 0)
			class_log, class_labels, _ = self.classifier(cat)
		else:
			CH_hidden, ner_tensor = self.M2_CNN(CH_BATCH, False)
			CH_hidden = torch.cat((CH_hidden, ner_tensor), -1)
			class_log, class_labels, _ = self.classifier(CH_hidden)
		'''
		else:
			CH_hidden = self.M2_CNN(CH_BATCH, train)
			class_log, class_labels, _ = self.classifier(CH_hidden)
		'''
		return diff_log, class_log, class_labels, CH_hidden
	def init_weight(self):
		initrange = 0.01
		#if config.CNN:
		if config.NER:
			self.ner_embeds.weight.data.uniform_(-initrange, initrange)
		for conv in self.M1_conv:
			init.xavier_uniform(conv.weight)
			init.constant(conv.bias, 0.1) 
		for conv in self.M2_conv:
			init.xavier_uniform(conv.weight)
			init.constant(conv.bias, 0.1)  
		#init.xavier_uniform(self.M2_fc1.weight)
		#init.constant(self.M2_fc1.bias, 0)  
		
		init.xavier_uniform(self.hidden2tag.weight)
		init.constant(self.hidden2tag.bias, 0.1)  
		init.xavier_uniform(self.hidden2diff.weight)
		init.constant(self.hidden2diff.bias, 0.1) 












