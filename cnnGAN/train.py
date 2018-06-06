#!/bin/python
# encoding: utf-8
import config
import sys
import time
import data_pro
import models

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as D
from focalloss import *
import os 
import numpy as np
import time
import pdb
import itertools

loss_function = FocalLoss(gamma=2)
datas = data_pro.dataSet()
datas.preprocess()
#datas.load_pkl()
#maxlen = config.maxlen
#torch.initial_seed(0)
model = models.GAN(config.tag_size, config.kernel_num, config.kernel_sizes1, config.kernel_sizes2)
model = model.cuda()
diff_distance = distance_loss()
gen_distance = gen_distance_loss()
parameters = itertools.ifilter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(model.parameters())
#Adadelta
M1_CNN_optimizer = optim.Adadelta(model.M1_conv.parameters())
M2_CNN_optimizer = optim.Adadelta(model.M2_conv.parameters())
diff_optimizer = optim.Adadelta(model.hidden2diff.parameters())
class_optimizer = optim.Adadelta(model.hidden2tag.parameters())
en_word_optimizer = optim.Adadelta(model.ENword_embeds.parameters())
en_dist_optimizer = optim.Adadelta(model.ENdist_embeds.parameters())
ch_word_optimizer = optim.Adadelta(model.CHword_embeds.parameters())
ch_dist_optimizer = optim.Adadelta(model.CHdist_embeds.parameters())
ner_optimizer = optim.Adadelta(model.ner_embeds.parameters())

#optimizer = optim.Adadelta(parameters)
#optimizer = optim.SGD(parameters, config.LR, momentum=0.8)
#optimizer = optim.Adam(parameters, config.lrn_rate)
maxprecision = 0.0
maxrecall = 0.0
maxF1 = 0.0
bestepoch = 0
final_label = []
golds = []

def data_unpack(cat_data, target_tensor, Cross=True):
	enmaxlen = config.EN_maxlen
	chmaxlen = config.CH_maxlen
	'''
	if EN:
		maxlen = config.EN_maxlen
	else:
		maxlen = config.CH_maxlen
	'''
	target_tensor = Variable(target_tensor).cuda()
	list_x = []
	if Cross:
		list_x = np.split(cat_data.numpy(), [enmaxlen, enmaxlen*2, enmaxlen*3, enmaxlen*3+2, enmaxlen*3+2+chmaxlen, enmaxlen*3+2+chmaxlen*2, enmaxlen*3+2+chmaxlen*3], 1)
		en_sent_tensor = Variable(torch.from_numpy(list_x[0])).cuda()
		en_dist1_tensor = Variable(torch.from_numpy(list_x[1])).cuda()
		en_dist2_tensor = Variable(torch.from_numpy(list_x[2])).cuda()
		ner_tensor = Variable(torch.from_numpy(list_x[3])).cuda()
		ch_sent_tensor = Variable(torch.from_numpy(list_x[4])).cuda()
		ch_dist1_tensor = Variable(torch.from_numpy(list_x[5])).cuda()
		ch_dist2_tensor = Variable(torch.from_numpy(list_x[6])).cuda()
		enbatch = (en_sent_tensor, en_dist1_tensor, en_dist2_tensor, ner_tensor)
		chbatch = (ch_sent_tensor, ch_dist1_tensor, ch_dist2_tensor, ner_tensor)
		return (enbatch, chbatch), target_tensor
	else:
		list_x = np.split(cat_data.numpy(), [chmaxlen, chmaxlen*2, chmaxlen*3], 1)
		ch_sent_tensor = Variable(torch.from_numpy(list_x[0])).cuda()
		ch_dist1_tensor = Variable(torch.from_numpy(list_x[1])).cuda()
		ch_dist2_tensor = Variable(torch.from_numpy(list_x[2])).cuda()
		ner_tensor = Variable(torch.from_numpy(list_x[3])).cuda()
		chbatch = (ch_sent_tensor, ch_dist1_tensor, ch_dist2_tensor, ner_tensor)
		return chbatch, target_tensor


def M1_CNN_step():
	M1_CNN_optimizer.step()
	en_dist_optimizer.step()
	ner_optimizer.step()
	en_word_optimizer.step()
def M2_CNN_step():
	M2_CNN_optimizer.step()
	ch_dist_optimizer.step()
	ner_optimizer.step()
	ch_word_optimizer.step()
def model_load():
	model.CHword_embeds.load_state_dict(torch.load(config.chword_emb_model))
	model.CHdist_embeds.load_state_dict(torch.load(config.chdist_emb_model))
	model.CHpos_embeds.load_state_dict(torch.load(config.chpos_emb_model))
	model.CHner_embeds.load_state_dict(torch.load(config.chner_emb_model))
	model.M2_conv.load_state_dict(torch.load(config.chconv_model))
	model.hidden2tag.load_state_dict(torch.load(config.h2tag_model))
def train():
	#model_load()

	

	train_vec = np.array(datas.train_vec, dtype=int)
	train_datasets = D.TensorDataset(data_tensor=torch.from_numpy(train_vec), 
									target_tensor=torch.from_numpy(datas.train_labels))
	train_dataloader = D.DataLoader(train_datasets, config.BATCH, True, num_workers=2)
	start = time.time()
	#pdb.set_trace()
	
	process = 0
	global golds
	now = time.time()
	class_labels = []
	target = []
	for epoch in range(config.epochs):
		print >> sys.stderr, "epoch{}/{}, epoch time {}".format(process, epoch, config.epochs, datas.timeSince(now))
		for (ex, ey) in train_dataloader:
			now = time.time()
			process += 1
			(enbatch, chbatch), target = data_unpack(ex, ey, True)
			batch = target.size()[0]
			if epoch < 25:
				
				model.zero_grad()
				diff_log, class_log, class_labels, _ = model(enbatch, chbatch, True)
				#pdb.set_trace()
				class_loss = F.cross_entropy(class_log[:batch], target)
				class_loss.backward()
				M1_CNN_step()
				class_optimizer.step()
			else:
				diff_tags = np.zeros(2*batch, dtype=int)
				diff_tags[:batch] = 1
				gen_tags = Variable(torch.from_numpy(np.ones(batch, dtype=int)).cuda())
				diff_tags = Variable(torch.from_numpy(diff_tags).cuda())
				count = 10
				for i in range(count):
					model.zero_grad()
					diff_log, class_log, class_labels, _ = model(enbatch, chbatch, True)
					diff_loss = F.cross_entropy(diff_log, diff_tags)
					diff_loss.backward()
					diff_optimizer.step()
				model.zero_grad()
				diff_log, class_log, class_labels, _ = model(enbatch, chbatch, True)
				gen_loss = F.cross_entropy(diff_log[batch:], gen_tags)
				
				#try:
				class_loss = F.cross_entropy(class_log[batch:], target)
				#except:
				#	pdb.set_trace()
				loss = config.C * gen_loss + class_loss
				loss.backward()
				class_optimizer.step()
				M2_CNN_step()
			#M2_CNN_step()
			
			'''
			##pdb.set_trace()
			if process %40 == 0:
				accuracy, precision, recall, F1 = calculate_accuracy(datas.getLabel(list(class_labels.data.cpu().numpy())), datas.getLabel(list(batch_tags.data.cpu().numpy())))
				print >> sys.stderr, "process {} epoch{}/{}, epoch time {}, loss{:.4}".format(process, epoch, config.epochs, datas.timeSince(now), loss.data)
				print >> sys.stderr, "accuracy:{:.4}, precision:{:.4}, recall:{:.4}, F1:{:.4}".format(accuracy, precision, recall, F1)
			'''
		accuracy, precision, recall, F1 = calculate_accuracy(datas.getLabel(list(class_labels.data.cpu().numpy())), datas.getLabel(list(target.data.cpu().numpy())))
		print >> sys.stderr, "epoch{}/{}, epoch time {}, loss{:.4}".format( epoch, config.epochs, datas.timeSince(now), class_loss.data[0])
		print >> sys.stderr, "accuracy:{:.4}, precision:{:.4}, recall:{:.4}, F1:{:.4}".format(accuracy, precision, recall, F1)
		test_labels, gold_labels = predict(datas.CH_test_vec, datas.CH_test_labels)
		#pdb.set_trace()
		precision, recall, F1 = datas.Score(datas.getLabel(test_labels), datas.getLabel(gold_labels), False)
		comparison(precision, recall, F1, datas.getLabel(test_labels), epoch)
		print >> sys.stderr, " precision:{:.4}, recall:{:.4}, F1:{:.4}".format( precision, recall, F1)
		if len(golds) == 0: golds = datas.getLabel(gold_labels)
		if epoch %20 == 0:
			print >> sys.stderr, "$$$$$########test data：epoch:{}, maxprecision:{:.4}, maxrecall:{:.4}, maxF1:{:.4}".format(bestepoch, maxprecision, maxrecall, maxF1)
		
		#losses = losses*1.0/(len(train_dataloader))
		#train_labels, gold_labels = predict(datas.train_vec, datas.train_labels)
		#pdb.set_trace()
	datas.Score(final_label, golds, True)
	#accuracy, precision, recall, F1 = calculate_accuracy(test_labels, gold_labels)
	print >> sys.stderr, "$$$$$########test data：epoch:{}, maxprecision:{:.4}, maxrecall:{:.4}, maxF1:{:.4}".format(bestepoch, maxprecision, maxrecall, maxF1)
	print >> sys.stderr, "training time {}".format(datas.timeSince(start))
	add = sys.argv[1]
	with open("result/test"+add+".labels", 'w') as f:
		space = ""
		for idx, label in enumerate(final_label):
			f.write(space+str(idx)+"\t"+label)
			space = "\n"
	with open("result/gold.labels", 'w') as f:
		space = ""
		for idx, label in enumerate(golds):
			f.write(space+str(idx)+"\t"+label)
			space = "\n"

def predict(data_vec, data_labels):
	maxlen = config.CH_maxlen
	dim = maxlen*3+2
	#if config.Wordnet:
	#	dim += 2
	batch = np.zeros((config.BATCH, dim), dtype=int)
	target = np.zeros(config.BATCH, dtype=int)
	index = 0
	out_labels = []
	for vec, label in zip(data_vec, data_labels):
		batch[index] = vec
		target[index] = label
		if (index+1)%config.BATCH == 0:
			epoch_tensor, target_tensor = data_unpack(torch.from_numpy(batch), torch.from_numpy(target), False)
			_, class_log, class_labels, _ = model([], epoch_tensor, False)
			out_labels.extend(class_labels.data.cpu().numpy())
		index = (index+1)%config.BATCH
	if index != 0:
		for i in range(index, config.BATCH):
			batch[i] = batch[0]
			target[i] = target[0]
		epoch_tensor, target_tensor = data_unpack(torch.from_numpy(batch), torch.from_numpy(target), False)
		_, class_log, class_labels, _ = model([], epoch_tensor, False)
		
		out_labels.extend(class_labels.data.cpu().numpy()[:index])
	#pdb.set_trace()
	return list(out_labels), data_labels

def calculate_accuracy(labels, gold_labels):
	tp = 0.0  #关系分类正确的总次数
	pos_pred = 0.0   #计算非other关系的正确数
	pos = 0.0 #非other关系实例总数
	right = 0.0
	#pdb.set_trace()
	for label, gold_label in zip(labels, gold_labels):
		#

		if gold_label == label:
			right += 1
		if label != datas.other_label:
				pos_pred += 1
		#pdb.set_trace()
		if gold_label != datas.other_label:
			pos += 1
			if gold_label == label:
				tp += 1
	if pos == 0 :
		pos = 0.01
	if pos_pred == 0:
		pos_pred = 0.01
	precision = tp / pos
	recall = tp / pos_pred
	m = precision+recall
	if m == 0:
		m = 0.001
	F1 = (2*precision*recall)/m
	accuracy = right / len(labels)
	return accuracy, precision, recall, F1

def comparison(precision, recall, F1, labels, epoch):
	global maxprecision
	global maxrecall
	global maxF1
	global final_label
	global bestepoch
	if maxF1 < F1:
		#torch.save(model.state_dict(), "result/"+sys.argv[1]+".model.state")
		#torch.save(model, "result/"+sys.argv[1]+".model")
		maxprecision = precision
		maxrecall = recall
		maxF1 = F1
		final_label = labels
		bestepoch = epoch

def configPrint(file):
	with open(file, 'r') as f:
		for line in f:
			line = line.strip()
			if line == "":
				continue
			print >> sys.stderr, line
	f.close()



if __name__ == '__main__':
	#torch.cuda.set_device(config.gpu_num)
	configPrint("config.py")
	train()

