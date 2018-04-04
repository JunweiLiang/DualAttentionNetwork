# coding=utf-8
# prepro for flickr30k, text matching


import sys,os,argparse,nltk,re
import cPickle as pickle
import numpy as np

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("splits")
	parser.add_argument("text")
	parser.add_argument("outpath")
	parser.add_argument("--noword2vec",action="store_true")
	parser.add_argument("--word2vec")
	parser.add_argument("--featpath")
	parser.add_argument("--noimgfeat",action="store_true")
	return parser.parse_args()
from collections import Counter

from tqdm import tqdm
def process_tokens(tokens):
	newtokens = []
	l = ("-","/", "~", '"', "'", ":","\)","\(","\[","\]","\{","\}")
	for token in tokens:
		# split then add multiple to new tokens
		newtokens.extend([one for one in re.split("[%s]"%("").join(l),token) if one != ""])
	return newtokens

def get_word2vec(args,word_counter):
	word2vec_dict = {}
	import io
	with io.open(args.word2vec, 'r', encoding='utf-8') as fh:
		for line in fh:
			array = line.lstrip().rstrip().split(" ")
			word = array[0]
			vector = list(map(float, array[1:]))
			if word in word_counter:
				word2vec_dict[word] = vector
			#elif word.capitalize() in word_counter:
			#	word2vec_dict[word.capitalize()] = vector
			#elif word.lower() in word_counter:
			#	word2vec_dict[word.lower()] = vector
			#elif word.upper() in word_counter:
			#	word2vec_dict[word.upper()] = vector

	#print "{}/{} of word vocab have corresponding vectors ".format(len(word2vec_dict), len(word_counter))
	return word2vec_dict

def prepro_each(args,data_type,imgids,img2txt,start_ratio=0.0,end_ratio=1.0):
	def word_tokenize(tokens):
		# nltk.word_tokenize will split ()
		# "a" -> '``' + a + "''"
		# lizzy's -> lizzy + 's
		# they're -> they + 're
		# then we remove and split "-"
		return process_tokens([token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)])

	word_counter,char_counter = Counter(),Counter() # lower word counter
	all_imgids = {}
	# generate (imgId,sents,sent_c) for data 
	data = []

	# for validation and testing
	sentids = []
	sentid2data = {}

	start_idx = int(round(len(imgids) * start_ratio))
	end_idx = int(round(len(imgids) * end_ratio))

	max_sent_length = 0
	max_word_size = 0

	for _,imgid in enumerate(tqdm(imgids)):
		
		all_imgids[imgid] = 1


		for sentid,sent in img2txt[imgid]:
			try:
				si = word_tokenize(sent.lower())
			except Exception as e:
				print "error for %s , %s"%(sent,e)
				print "remove non-ascii and try again"
				sent = ''.join([i if ord(i) < 128 else ' ' for i in sent])
				print "New sent:%s"%sent
				si = word_tokenize(sent.lower())
			max_sent_length = max(max_sent_length,len(si))
			csi = [list(sij) for sij in si]

			for sij in si:
				word_counter[sij]+=1
				max_word_size = max(max_word_size,len(sij))
				for sijk in sij:
					char_counter[sijk]+=1
			
			data.append((imgid,si,csi))
			sentids.append(sentid)
			# for validation and testing
			sentid2data[sentid] = {"sents":si,"sents_c":csi}

		

		# for validation and testing, 
		# this will make the file too big, will make in the model training
		"""
		if data_type != "train":
			# also need to add all other sentence for each image
			for imgid_other in imgids:
				if imgid_other == imgid:
					continue

				for idx,sent in img2txt[imgid_other]:
					try:
						si = word_tokenize(sent.lower())
					except Exception as e:
						print "error for %s , %s"%(sent,e)
						print "remove non-ascii and try again"
						sent = ''.join([i if ord(i) < 128 else ' ' for i in sent])
						print "New sent:%s"%sent
						si = word_tokenize(sent.lower())
					max_sent_length = max(max_sent_length,len(si))
					csi = [list(sij) for sij in si]

					for sij in si:
						word_counter[sij]+=1
						max_word_size = max(max_word_size,len(sij))
						for sijk in sij:
							char_counter[sijk]+=1
					
					data.append((imgid,si,csi))
					idxs.append(idx)
		"""

	imgid2feat = {}
	if not args.noimgfeat:		
		for imgid in all_imgids:
			featpath = os.path.join(args.featpath,imgid+".npy")
			feat = np.load(featpath)
			imgid2feat[imgid] = feat

		print "saving image feature"
		with open(os.path.join(args.outpath,"%s_imgid2feat.npz"%data_type), "w") as f:
			np.savez_compressed(f, **imgid2feat)
	

	d = {
		'data':data, # (imgid,si,csi)
		"sentids":sentids, # (sentid)
		"imgids":imgids,
		#"sentid2data":sentid2data
	}
	word2vec = {}
	if args.noword2vec:
		shared = {
			#"imgid2feat":imgid2feat, # save the image feature to a separete npz file
			"word_counter":word_counter,
			"char_counter":char_counter,
			"word2vec":{},
		}
	else:
		word2vec = get_word2vec(args,word_counter) # all word => vec
		shared = {
			#"imgid2feat":imgid2feat,
			"word_counter":word_counter,
			"char_counter":char_counter,
			"word2vec":word2vec,
		}

	print "data:%s,sentids:%s,imgids:%s max sentence length:%s,max word size:%s, char entry:%s, word entry:%s,word2vec entry:%s,imgfeat:%s"%(len(data),len(sentids),len(imgids),max_sent_length,max_word_size,len(char_counter),len(word_counter),len(word2vec),len(imgid2feat))

	pickle.dump(d,open(os.path.join(args.outpath,"%s_data.p"%data_type),"wb"))
	pickle.dump(shared,open(os.path.join(args.outpath,"%s_shared.p"%data_type),"wb"))

	# save the image feature to a separate npz file, otherwise the pickle will be too lartge





def getIds(args):
	trainpath,valpath,testpath = os.path.join(args.splits,"train.lst"),os.path.join(args.splits,"val.lst"),os.path.join(args.splits,"test.lst")

	trainIds = [os.path.splitext(os.path.basename(line.strip()))[0] for line in open(trainpath,"r").readlines()]
	valIds = [os.path.splitext(os.path.basename(line.strip()))[0] for line in open(valpath,"r").readlines()]

	testIds = [os.path.splitext(os.path.basename(line.strip()))[0] for line in open(testpath,"r").readlines()]

	print "total trainId:%s,val:%s, test:%s images"%(len(trainIds),len(valIds),len(testIds))

	return trainIds,valIds,testIds



if __name__ == "__main__":
	args = get_args()

	# load the imgId to text
	imgid2text = {}

	with open(args.text,"r") as f:
		for line in f:
			image,text = line.strip().split("\t")
			sentid = image.strip() 
			imageId = image.strip().split(".")[0]
			if not imgid2text.has_key(imageId):
				imgid2text[imageId] = []

			imgid2text[imageId].append((sentid,text.strip()))

	print "got %s images for text file"%(len(imgid2text))

	if not os.path.exists(args.outpath):
		os.makedirs(args.outpath)

	trainIds,valIds,testIds = getIds(args)

	prepro_each(args,"train",trainIds,imgid2text,0.0,1.0)
	prepro_each(args,"val",valIds,imgid2text,0.0,1.0)
	prepro_each(args,"test",testIds,imgid2text,0.0,1.0)

