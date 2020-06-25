from __future__ import print_function, division
import sys, os, argparse, gensim, sklearn, pickle
from thyme import read_thyme_documents, write_entities_to_anafora, read_txt_corpus
from tokenization import find_alternative
from features import EventFeatureHandler, Timex3FeatureHandler
from sklearn import linear_model
from sklearn import svm
from evaluation import  evaluate
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier 
import numpy as np
sys.setrecursionlimit(50000)

__author__ = "Artuur Leeuwenberg"
__email__ = "tuur.leeuwenberg@cs.kuleuven.be"

parser = argparse.ArgumentParser(description='Implementation for a simple SVM model for EVENT and TIMEX3 training and prediction.')
parser.add_argument('thyme', type=str,
                    help='thyme corpus')	
parser.add_argument('-docs', type=int,default=100000000000,
                    help='maximum number of documents used in each set (train, dev)')
parser.add_argument('-save_document_structure', type=str, default=None,
                    help='Saves input document structure (with tokenization and POS).')
parser.add_argument('-load_document_structure', type=str, default=None,
                    help='Loads input document structure (with tokenization and POS).')
parser.add_argument('-lowercase', type=int, default=1,
                    help='Consider all text as lowercased. default:1')
parser.add_argument('-conflate_digits', type=int, default=1,
                    help='Conflates all digits. default:1')
parser.add_argument('-ctakes_out_dir', type=str,
                    help='Use cTAKES output features (POS and/or dependency path)')	
parser.add_argument('-pos', type=int, default=1,
                    help='Using POS features (default=1)')																					
parser.add_argument('-timex_n', type=int, default=6,
                    help='Size of token-ngrams used as timex3 candidates')	
parser.add_argument('-train_models', type=str, default=None,
                    help='Saves the trained models to a directory')	
parser.add_argument('-test_models', type=str, default=None,
                    help='test the models in the provided directory')	
parser.add_argument('-model', type=str, default='svm',
                    help='Type of model used: svm or logit')																				
parser.add_argument('-to_anafora', type=str, default='xml_output',
                    help='Output directory for anafora xml predictions')
parser.add_argument('-test_xml', type=int, default=1,
                    help='Ground truth xml is available for the test set (default=1)')
parser.add_argument('-unk_token', type=int, default=1,
                    help='Use UNK token for words with counts lower or equal than X (default:3)')
parser.add_argument('-unsup_corpus', type=str,
                    help='Provide an unannotated text corpus')																															
parser.add_argument('-word_replacement_embeddings', type=str,
                    help='Use word embeddings to replace unknown words during prediction time. (provide word vectors in w2v .bin format)')	
args = parser.parse_args()


if args.load_document_structure:
	with open(args.load_document_structure, 'rb') as f:
		thyme_document_structure = pickle.load(f)[:args.docs]
else:
	thyme_document_structure = read_thyme_documents(args.thyme, regex='.*Temp.*', max_documents=args.docs, closure=[], lowercase=bool(args.lowercase), conflate_digits=bool(args.conflate_digits), ctakes_out_dir=args.ctakes_out_dir,pos=args.pos, less_strict=not(bool(args.test_xml)))	
	if args.save_document_structure:
		if not os.path.isdir(os.path.dirname(args.save_document_structure)):
			os.makedirs(os.path.dirname(args.save_document_structure))
		with open(args.save_document_structure, 'wb') as f:
			pickle.dump(thyme_document_structure, f)


if args.unsup_corpus:
	unsup_corpus  = read_txt_corpus(args.unsup_corpus)
	unsup_vocab = {}
	tokens = []
	for i,doc in enumerate(unsup_corpus):
		doc_vocab = doc.tokenization.get_vocabulary()
		tokens += [[t.string for t in doc.tokenization.tokens]]
		for k,v in doc_vocab.items():
			if k in unsup_vocab:
				unsup_vocab[k] += v
			else:
				unsup_vocab[k] = v	


if args.train_models:
	feature_extractor_e = EventFeatureHandler()
	feature_extractor_t = Timex3FeatureHandler()

	label_encoder_e = preprocessing.LabelEncoder()
	label_encoder_pol = preprocessing.LabelEncoder()
	label_encoder_type = preprocessing.LabelEncoder()
	label_encoder_degree = preprocessing.LabelEncoder()
	label_encoder_modality = preprocessing.LabelEncoder()

	label_encoder_t = preprocessing.LabelEncoder()
	label_encoder_ttype = preprocessing.LabelEncoder()

	vocabulary = {}
	for i,doc in enumerate(thyme_document_structure):
			
		doc_vocab = doc.tokenization.get_vocabulary()
		for k,v in doc_vocab.items():
			if k in vocabulary:
				vocabulary[k] += v
			else:
				vocabulary[k] = v
	
	if args.unsup_corpus:

		for k,v in vocabulary.items():
			if v < args.unk_token:
				continue
			if not k in unsup_vocab:
				vocabulary[k] = 0
	
	if args.model == 'logit':
		classifier_e =  OneVsRestClassifier(linear_model.LogisticRegression(), n_jobs=4)
		classifier_t = OneVsRestClassifier(linear_model.LogisticRegression(), n_jobs=4)
	elif args.model == 'svm':
		classifier_e =  OneVsRestClassifier(svm.LinearSVC(C=.1), n_jobs=4)
		classifier_t =  OneVsRestClassifier(svm.LinearSVC(C=.1), n_jobs=4)
		classifier_ttype =  OneVsRestClassifier(svm.LinearSVC(C=.1), n_jobs=4)
		classifier_pol = OneVsRestClassifier(svm.LinearSVC(C=.1), n_jobs=4) 
		classifier_type = OneVsRestClassifier(svm.LinearSVC(C=.1), n_jobs=4) 
		classifier_degree = OneVsRestClassifier(svm.LinearSVC(C=.1), n_jobs=4) 
		classifier_modality = OneVsRestClassifier(svm.LinearSVC(C=.1), n_jobs=4) 
		
	else:
		print('no such model')
		exit()
		
	
elif args.test_models:
	with open(args.test_models + '/event_spans.p', 'rb') as f:
		feature_extractor_e, label_encoder_e, classifier_e, label_encoder_pol, classifier_pol, label_encoder_type, classifier_type, label_encoder_degree, classifier_degree, label_encoder_modality, classifier_modality,  vocabulary = pickle.load(f)
		
	with open(args.test_models + '/timex3_spans.p', 'rb') as f:
		feature_extractor_t, label_encoder_t, classifier_t, label_encoder_ttype, classifier_ttype, vocabulary = pickle.load(f)
		
		
	if args.word_replacement_embeddings:
		print('reading embeddings',args.word_replacement_embeddings)
		embeddings = gensim.models.Word2Vec.load_word2vec_format(args.word_replacement_embeddings, binary=True) 	
		

else:
	print('ERROR: either test, or train models!')
	exit()
		
# ----------------------- Constructing Data Set

Xe, Ye = [], []
Xt, Yt = [], []

print('extracting features from', len(thyme_document_structure), 'documents')
for i,doc in enumerate(thyme_document_structure):	

			
	if args.word_replacement_embeddings and args.test_models:
		doc.tokenization.adaptation_replace(vocabulary, embeddings=embeddings)


	if args.unk_token:
		doc.tokenization.adaptation_unk(vocabulary, unk_threshold=args.unk_token)		
	elif args.unsup_corpus:
		doc.tokenization.adaptation_unk(vocabulary, unk_threshold=0)

	X_e = doc.get_event_candidates()
	X_t = doc.get_timex3_candidates(size=args.timex_n)	
			
		
	for e in X_e:
		e.phi = feature_extractor_e.extract(e, doc, update=args.train_models)

	for t in X_t:
		t.phi = feature_extractor_t.extract(t, doc, update=args.train_models)		
					
	Y_e = [e.type for e in X_e]
	Y_t = [t.type for t in X_t]

	Xe.append(X_e)	
	Xt.append(X_t)	
	
	Ye.append(Y_e)
	Yt.append(Y_t)
	


if args.train_models:		
	feature_extractor_e.update_vectorizer()
	feature_extractor_t.update_vectorizer()

	print ('fitting event classifier')
	X = feature_extractor_e.vectorize([e.phi for x_e in Xe for e in x_e])
	classifier_e.fit(X, label_encoder_e.fit_transform([e for y_e in Ye for e in y_e]))
	
	events = [e for x_e in Xe for e in x_e if e.type == 'EVENT']

	
	events_X = feature_extractor_e.vectorize([e.phi for e in events])
	classifier_pol.fit(events_X, label_encoder_pol.fit_transform([e.attributes['EVENT_Polarity'] for e in events]))
	classifier_type.fit(events_X, label_encoder_type.fit_transform([e.attributes['EVENT_Type'] for e in events]))
	classifier_degree.fit(events_X, label_encoder_degree.fit_transform([e.attributes['EVENT_Degree'] for e in events]))
	classifier_modality.fit(events_X, label_encoder_modality.fit_transform([e.attributes['EVENT_CONTEXTUAL_MODALITY'] for e in events]))
	
	if not os.path.isdir(os.path.dirname(args.train_models)):
			os.makedirs(os.path.dirname(args.train_models))
	with open(args.train_models + '/event_spans.p', 'wb') as f:
		pickle.dump([feature_extractor_e, label_encoder_e, classifier_e, label_encoder_pol, classifier_pol, label_encoder_type, classifier_type, label_encoder_degree, classifier_degree, label_encoder_modality, classifier_modality,  vocabulary],f)


	print ('fitting timex3 classifier')
	classifier_t.fit(feature_extractor_t.vectorize([t.phi for x_t in Xt for t in x_t]), label_encoder_t.fit_transform([t for y_t in Yt for t in y_t]))
	timex3 = [t for x_t in Xt for t in x_t if t.type == 'TIMEX3']
	timex3_X = feature_extractor_t.vectorize([t.phi for t in timex3])
	classifier_ttype.fit(timex3_X, label_encoder_ttype.fit_transform([t.attributes['TIMEX_Class'] for t in timex3]))

	
	with open(args.train_models + '/timex3_spans.p', 'wb') as f:
		pickle.dump([feature_extractor_t, label_encoder_t, classifier_t, label_encoder_ttype, classifier_ttype, vocabulary],f)

elif args.test_models:
	print('predicting...')
	candidates_e = [e for x_e in Xe for e in x_e]
	
	e_vecs = feature_extractor_e.vectorize([e.phi for e in candidates_e])
	predicted_e = list(label_encoder_e.inverse_transform(classifier_e.predict(e_vecs)))
	

	event_vecs = feature_extractor_e.vectorize([e.phi for i,e in enumerate(candidates_e) if predicted_e[i] == 'EVENT'])
	predicted_pol = list(label_encoder_pol.inverse_transform(classifier_pol.predict(event_vecs)))
	predicted_type = list(label_encoder_type.inverse_transform(classifier_type.predict(event_vecs)))
	predicted_degree = list(label_encoder_degree.inverse_transform(classifier_degree.predict(event_vecs)))
	predicted_modality = list(label_encoder_modality.inverse_transform(classifier_modality.predict(event_vecs)))
	
	attribute_index = 0
	for i, entity_candidate in enumerate(candidates_e):
		if predicted_e[i] == 'EVENT':
			entity_candidate.attributes['Polarity'] = predicted_pol[attribute_index]
			entity_candidate.attributes['Type'] = predicted_type[attribute_index]
			entity_candidate.attributes['Degree'] = predicted_degree[attribute_index]
			entity_candidate.attributes['Modality'] = predicted_modality[attribute_index]
			attribute_index += 1
	
	if args.test_xml:
		Ytrue = [e for y_e in Ye for e in y_e]
		evaluate(predicted_e, Ytrue, transform=False)


	candidates_t = [t for x_t in Xt for t in x_t]

	predicted_t = list(label_encoder_t.inverse_transform(classifier_t.predict(feature_extractor_t.vectorize([t.phi for x_t in Xt for t in x_t]))))
	timex3_vecs = feature_extractor_t.vectorize([t.phi for i,t in enumerate(candidates_t) if predicted_t[i] == 'TIMEX3'])
	predicted_ttype = list(label_encoder_ttype.inverse_transform(classifier_ttype.predict(timex3_vecs)))
	attribute_index = 0
	for i, timex3_candidate in enumerate(candidates_t):
		if predicted_t[i] == 'TIMEX3':
			timex3_candidate.attributes['Class'] = predicted_ttype[attribute_index]
			attribute_index += 1	


	if args.test_xml:
		Ytrue = [t for y_t in Yt for t in y_t]
		evaluate(predicted_t, Ytrue, transform=False)	
	
	if args.to_anafora:
		write_entities_to_anafora(candidates_e, predicted_e,[t for x_t in Xt for t in x_t], predicted_t, args.to_anafora)	
	
else:
	print('ERROR: either test, or train models!')
	exit()
		
