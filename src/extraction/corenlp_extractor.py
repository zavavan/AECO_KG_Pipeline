# command to run the stanford core nlp server
# nohup java -mx8g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9050 -timeout 15000 >/dev/null 2>&1 &
from collections import defaultdict

from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords
import multiprocessing as mp
import networkx as nx
import itertools
import json
import nltk
import sys
import ast
import os
import re
from tqdm import tqdm
import Levenshtein
from pathlib import Path

from functools import partial


dataset_dump_dir = '../../dataset/aeco/'
dygiepp_output_dump_dir = '../../outputs/dygiepp_output/'
llm_output_dump_dir = '../../outputs/llm_output/'

output_dir = '../../outputs/extracted_triples/'
if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Folder created: {output_dir}")
else:
	print(f"Folder already exists: {output_dir}")

acro_output_dir = '../../outputs/extracted_triples/acronyms/'
debug_output_dir = '../../outputs/extracted_triples/debug/'

if not os.path.exists(acro_output_dir):
	os.makedirs(acro_output_dir)
	print(f"Folder created: {acro_output_dir}")
else:
	print(f"Folder already exists: {acro_output_dir}")

if not os.path.exists(debug_output_dir):
	os.makedirs(debug_output_dir)
	print(f"Folder created: {debug_output_dir}")
else:
	print(f"Folder already exists: {debug_output_dir}")
stops = list(stopwords.words('english')) + ['it', 'we', 'they', 'its']

# used to check if an openie entity is a real entity of the domain
def checkEntity(e, e_list):
	eresult = None
	if e not in stops:
		for ei in e_list:
			if e in ei or ei in e:
				eresult = ei
	return eresult


def findTokens(s, tokens):
	for i in range(len(s)):
		try:
			if s[i : i + len(tokens)] == tokens:
				return i , i + len(tokens)
		except:
			return -1,-1
	return -1,-1


def pairwise(iterable):
	it = iter(iterable)
	a = next(it, None)
	for b in it:
		yield (a, b)
		a = b


def mapEntityAcronyms(acronyms, e):
	if e.lower() in acronyms:
		return acronyms[e.lower()]
	else:
		return e


def detectAcronyms(elist):
	acronyms = {}
	regex_acronym = re.compile("\(.*")
	## building integrated photovoltaic (BIPV)
	for e in elist:
		e_cleaned_without_acr = regex_acronym.sub('', e).strip().lower()
		base_acronym = ''.join([token[0] for token in nltk.word_tokenize(e_cleaned_without_acr)])
		potential_acrs = [ acr.replace('( ', '').replace(' )', '').replace('(', '').replace(')', '').lower()  for acr in regex_acronym.findall(e)]
		for acr in potential_acrs:
			if acr == base_acronym:
				acronyms[acr] = e_cleaned_without_acr
				acronyms[e] = e_cleaned_without_acr
	return acronyms


# Function to compute similarity
def compute_levenshtein_similarity(string1, string2):
	# Compute the Levenshtein distance
	distance = Levenshtein.distance(string1, string2)
	# Normalize similarity: higher score means more similar
	max_len = max(len(string1), len(string2))
	similarity = 1 - (distance / max_len) if max_len > 0 else 1  # Avoid division by zero
	return similarity


def detectAcronymsLenient(elist):
	acronyms = {}
	regex_acronym = re.compile("\(.*\)")
	regex_acronym_followed = re.compile("(\(.*\))(.+)")

	## building integrated photovoltaic (BIPV)
	for e in elist:
		e_cleaned_without_acr = regex_acronym.sub('', e)
		e_cleaned_without_acr = re.sub(r'\s+', ' ', e_cleaned_without_acr).strip().lower()

		base_acronym = ''.join([token[0] for token in nltk.word_tokenize(e_cleaned_without_acr)])
		potential_acrs = [acr.replace('( ', '').replace(' )', '').replace('(', '').replace(')', '').lower() for acr in
						  regex_acronym.findall(e)]
		for acr in potential_acrs:
			if compute_levenshtein_similarity(acr, base_acronym) > .5:
				if regex_acronym_followed.search(e):
					acronyms[acr] = regex_acronym_followed.sub('', e).strip().lower()
				else:
					acronyms[acr] = e_cleaned_without_acr

				acronyms[e.lower()] = e_cleaned_without_acr
	return acronyms

def getDygieppResults(dresult):
	sentences = dresult['sentences']
	#print('num sentences: ' + str(len(sentences)))
	dner = dresult['predicted_ner']
	#print('num predicted_ner: ' + str(len(dner)))
	drelations = dresult['predicted_relations']
	#print('num predicted_relations: ' + str(len(drelations)))

	text = [token for sentence in sentences for token in sentence]
	sentence2data = {}

	for i in range(len(sentences)):
		entities = []
		relations = []
		for ner_el in dner[i]:
			e = ' '.join(text[ner_el[0]:ner_el[1]+1])
			e_type = ner_el[2]
			entities += [(e, e_type)]
		for relations_el in drelations[i]:
			r = relations_el[4]
			#if r == 'CONJUNCTION':
			#	continue
			e1 = ' '.join(text[relations_el[0]:relations_el[1]+1])
			e2 = ' '.join(text[relations_el[2]:relations_el[3]+1])
			relations += [(e1, r, e2)]

		sentence2data[i] = {'entities' :  entities, 'relations' : relations}
	return sentence2data


## This is the function that reads LLM-generated ent/rels from dygiepp pre-formatted file
# It discards overlapping entities and entities with entity token spans > 7 and their corresponding relations:


def getLlmResults(dresult):
	print(dresult['doc_key'])
	sentences = dresult['sentences']
	#print('num sentences: ' + str(len(sentences)))
	dner = dresult['predicted_ner']
	#print('num predicted_ner: ' + str(len(dner)))
	drelations = dresult['predicted_relations']
	#print('num predicted_relations: ' + str(len(drelations)))

	text = [token for sentence in sentences for token in sentence]
	sentence2data = {}
	added_ent_count = 0
	added_rel_count = 0
	discarded_ent_count=0
	discarded_rel_count = 0
	for i in range(len(sentences)):
		entities = []
		relations = []
		last_end = -1  # Track the last included entity's end index
		candidate_entities = dner[i]

		# Sort entities by start index (n1), then by length (longest first)
		candidate_entities.sort(key=lambda x: (x[0], -(x[1] - x[0])))
		# discard entities overalpping or with more than 6 tokens
		for ner_el in candidate_entities:
			if int(ner_el[1]) - int(ner_el[0]) > 6 or ner_el[0] < last_end :
				discarded_ent_count+=1
			else:
				e = ' '.join(text[ner_el[0]:ner_el[1]+1])
				e_type = ner_el[2]
				entities += [(e, e_type)]
				added_ent_count+=1
				last_end = ner_el[1]  # Update last_end to this entity's end
		entity_matches = [ent[0] for ent in entities]
		for relations_el in drelations[i]:
			if (' '.join(text[relations_el[0]:relations_el[1]+1]) not in entity_matches)  or  (' '.join(text[relations_el[2]:relations_el[3]+1]) not in entity_matches) :
				discarded_rel_count+=1
			else:
				r = relations_el[4]
				e1 = ' '.join(text[relations_el[0]:relations_el[1]+1])
				e2 = ' '.join(text[relations_el[2]:relations_el[3]+1])
				relations += [(e1, r, e2)]
				added_rel_count+=1

		sentence2data[i] = {'entities' :  entities, 'relations' : relations}

	## discard relations with entity token spans > 7 (it doesn't alter remove entities):
	## not needed anymore as it is done in the loop above
	#discarded_ent_count=0
	#discarded_rel_count = 0
	#for i,entRel in sentence2data.items():
	#	for rel in entRel['relations']:
	#		if len(rel[0].split())>7 or len(rel[2].split())>7:
	#			entRel['relations'].remove(rel)

	#print(f"Imported {added_ent_count} entities from LLM")
	#print(f"Imported {added_rel_count} relations from LLM")
	if discarded_ent_count > 0:
		print(f"Discarded {discarded_ent_count} long entities from LLM")
	if discarded_rel_count > 0:
		print(f"Discarded {discarded_rel_count} relations connecting long entities from LLM")

	return sentence2data



'''def solveCoref(corenlp_out):
	tokens = [token for sentence in corenlp_out['sentences'] for token in sentence['tokens']]
	print([t['lemma'] for t in tokens])

	for coref_num in corenlp_out['corefs']:
		representative_sentence = None
		representative_span = None
		coref_list = corenlp_out['corefs'][coref_num]
		print(coref_num)
		tokenSpan2lemma = {}
		for el in coref_list:
			if el['isRepresentativeMention'] == True:
				#print('Sentence:')
				#print(' '.join([t['originalText'] for t in corenlp_out['sentences'][el['sentNum']+1]['tokens']]))
				represenative_tokens = tokens[el['startIndex']:el['endIndex']]
				represenative_lemma = [t['lemma'] for t in represenative_tokens]
				print('Representative\n', represenative_lemma, el['startIndex'], el['endIndex'])
		
		for el in coref_list:
			if el['isRepresentativeMention'] == False:
				tokenSpan2lemma[el['startIndex'], el['endIndex']] = represenative_lemma
				#print((el['startIndex'], el['endIndex']))
				other_tokens = tokens[el['startIndex']:el['endIndex']]
				other_lemma = [t['lemma'] for t in other_tokens]
				print('others\n', other_lemma, el['startIndex'], el['endIndex'])
	print('-----------------------------------------------')
'''


def getOpenieTriples(corenlp_out, dygiepp, cso_topics,acronyms):
	relations = []
	for i in range(len(corenlp_out['sentences'])): #sentence in corenlp_out['sentences']:
		sentence = corenlp_out['sentences'][i]
		openie = sentence['openie']

		if i < len(dygiepp.keys()):
			dygiepp_sentence_entities = [x for (x, xtype) in dygiepp[i]['entities']]
			#print(dygiepp_sentence_entities)


		for el in openie:
			#print(el)
			subj = el['subject']
			obj = el['object']
			relation_token_numbers = el['relationSpan']

			#check if the relation is a verb
			relation_tokens = [t['lemma'] for t in sentence['tokens'] \
					if t['index'] > relation_token_numbers[0] and \
						t['index'] <= relation_token_numbers[1]  and \
						t['pos'].startswith('VB') ]

			# check if there is a passive
			relation = None
			passive = False
			if relation_tokens != []:
				if len(relation_tokens) == 1:
					relation = relation_tokens[0]
				else:
					if relation_tokens[-2] == 'be': #passive
						passive = True
					relation = relation_tokens[-1]

			#check on subject and obejct. They must exist as entities
			checked_subj = checkEntity(subj, dygiepp_sentence_entities + cso_topics)
			checked_obj = checkEntity(obj, dygiepp_sentence_entities + cso_topics)

			if checked_subj is not None and checked_obj is not None and relation is not None:
				checked_subj = mapEntityAcronyms(acronyms, checked_subj)
				checked_obj = mapEntityAcronyms(acronyms, checked_obj)
				if not passive:
					#print((checked_subj, relation, checked_obj))
					relations += [(checked_subj, relation, checked_obj, i)]
				else:
					#print((checked_obj, relation, checked_subj), '#passive')
					relations += [(checked_obj, relation, checked_subj, i)]
	return set(relations)


def getPosTriples(corenlp_out, dygiepp, cso_topics, acronyms):
	triples = []
	for i in range(len(corenlp_out['sentences'])): 
		sentence = corenlp_out['sentences'][i]
		sentence_tokens_text = [t['originalText'] for t in sentence['tokens']]
		sentence_tokens_text_lemma = [t['lemma'] for t in sentence['tokens']]
		sentence_tokens_pos = [t['pos'] for t in sentence['tokens']]

		if i < len(dygiepp.keys()):
			dygiepp_sentence_entities = [x for (x, xtype) in dygiepp[i]['entities']]


		entities_in_sentence = []
		for e in set(dygiepp_sentence_entities + cso_topics):
			start, end = findTokens(sentence_tokens_text, nltk.word_tokenize(e))
			if start != -1:
				entities_in_sentence += [(start, end)]

		#check verbs between each pair of entities
		for ((starti, endi), (startj, endj)) in itertools.combinations(entities_in_sentence, 2):
			ei = ' '.join(sentence_tokens_text[starti:endi])
			ej = ' '.join(sentence_tokens_text[startj:endj])
			ei = mapEntityAcronyms(acronyms, ei)
			ej = mapEntityAcronyms(acronyms, ej)

			verb_relations = []
			if endi < startj and startj - endi <= 10:
				verb_pattern = ''
				for k, pos in enumerate(sentence_tokens_pos[endi + 1:startj]):
					sentence_tokens_text_window = sentence_tokens_text_lemma[endi + 1:startj]
					if 'VB' in pos:
						verb_pattern += ' ' + sentence_tokens_text_window[k]
					elif verb_pattern != '':
						verb_relations += [verb_pattern.strip()]
						triples += [(ei, verb_pattern.strip(), ej, i)]
						verb_pattern = ''

			elif endj < starti and starti - endj <= 10:
				verb_pattern = ''
				for k, pos in enumerate(sentence_tokens_pos[endj + 1:starti]):
					sentence_tokens_text_window = sentence_tokens_text_lemma[endj + 1:starti]
					if 'VB' in pos:
						verb_pattern += ' ' + sentence_tokens_text_window[k]
					elif verb_pattern != '':
						verb_relations += [verb_pattern.strip()]
						triples += [(ej, verb_pattern.strip(), ei, i)]
						verb_pattern = ''

	# managing passive
	new_triples = []
	for (s,p,o,i) in triples:
		p_tokens = nltk.word_tokenize(p)
		v = p_tokens[-1]
		if 'be' in p_tokens[:-1] and len(p_tokens) > 1:
			new_triples += [(o, v, s, i)]
			#print((s,p,o), 'PASSIVE->', (o,v,s))
		else:
			new_triples += [(s,v,o, i)]
			#print((s,p,o), '->', (s,v,o))
	return set(new_triples)


def getDependencyTriples(corenlp_out, dygiepp, cso_topics, acronyms):
	triples = []
	validPatterns = [
			('nsubj', 'obj'), 
			('acl:relcl', 'obj'), 
			('nsubj', 'obj', 'conj'), 
			('conj', 'obl', 'nsubj:pass'), 
			('acl', 'obj'), 
			('nmod', 'nsubj', 'obj'),
			('obl', 'nsubj:pass'),
			('nsubj', 'obj', 'nmod'),
			('acl:relcl', 'obl'),
			('obl', 'acl'),
			('nmod', 'obj', 'acl'),
			('acl', 'obj', 'nmod'),
		]
	
	for i in range(len(corenlp_out['sentences'])): 
		sentence = corenlp_out['sentences'][i]
		sentence_tokens_text = [t['originalText'] for t in sentence['tokens']]
		dependencies = sentence['basicDependencies']
		tokens = sentence['tokens']
		
		#graph creation
		g = nx.Graph()
		for dep in dependencies:
			governor_token_number = dep['governor']
			dependent_token_number = dep['dependent']
			g.add_node(governor_token_number, postag=tokens[governor_token_number - 1]['pos'], text=tokens[governor_token_number - 1]['lemma'])
			g.add_node(dependent_token_number, postag=tokens[dependent_token_number - 1]['pos'], text=tokens[governor_token_number - 1]['lemma'])
			g.add_edge(governor_token_number, dependent_token_number, label=dep['dep'])

		if i < len(dygiepp.keys()):
			dygiepp_sentence_entities = [x for (x, xtype) in dygiepp[i]['entities']]

		#finidng position of entities as token numbers
		entities_in_sentence = []
		for e in set(dygiepp_sentence_entities + cso_topics):
			start, end = findTokens(sentence_tokens_text, nltk.word_tokenize(e))
			if start != -1:
				entities_in_sentence += [(start, end)]

		# check path between entity pairs
		for ((starti, endi), (startj, endj)) in itertools.combinations(entities_in_sentence, 2):
			ei = ' '.join(sentence_tokens_text[starti:endi])
			ej = ' '.join(sentence_tokens_text[startj:endj])
			ei = mapEntityAcronyms(acronyms, ei)
			ej = mapEntityAcronyms(acronyms, ej)
			paths = list(nx.all_simple_paths(G=g, source=endi, target=endj, cutoff=4))

			# check if there are verbs
			for path in paths:
				# dependencies path
				path_dep = [g.edges()[a,b]['label'] for (a,b) in pairwise(path)]
				v = False
				verbs = ''
				for node in path:
					if g.nodes[node]['postag'].startswith('VB'):
						v = True
						verbs += g.nodes[node]['text'] + ' '
				verbs = verbs.strip()

				if tuple(path_dep) in validPatterns and v:
					#if tuple(path_dep) == ('nsubj:pass', 'obl', 'conj'):
					#	print(sentence_tokens_text)
					#	print((ei, verbs, ej))
					triples += [(ei, verbs, ej, i)]
				elif tuple(path_dep)[::-1] in validPatterns and v:
					#if tuple(path_dep)[::-1]  == ('nsubj:pass', 'obl', 'conj'):
					#	print(sentence_tokens_text)
					#	print((ei, verbs, ej))
					triples += [(ej, verbs, ei, i)]
	return set(triples)

# this is used to map entities to acronyms and prepare the list of relations of each file that can be saved
# similarly to the other extractor tools
def manageEntitiesAndDygieepRelations(dygiepp, llm, entities, acronyms):
	dygiepp_entities = []
	llm_entities = []
	dygiepp_e2type = {}
	llm_e2type = {}
	e2type = {}
	ok_entities = []
	ok_relations_dygiepp = []
	ok_relations_llm = []

	for sentence in dygiepp:
		for (x, xtype) in dygiepp[sentence]['entities']:
			dygiepp_e2type[x] = xtype
	for sentence in llm:
		for (x, xtype) in llm[sentence]['entities']:
			llm_e2type[x] = xtype

	dygiepp_ent_num = len(dygiepp_e2type)

	dygiepp_e2type.update(llm_e2type)

	print('added llm entities =  ' + str(len(dygiepp_e2type) - dygiepp_ent_num))
	e2type = dygiepp_e2type

	for e in entities:
		if e in e2type:
			ok_entities += [(mapEntityAcronyms(acronyms, e), e2type[e])]
		else:
			ok_entities += [(mapEntityAcronyms(acronyms, e), 'CSO Topic')]
	ok_entities = set(ok_entities)

	for sentence in dygiepp:
		ok_relations_dygiepp += [(mapEntityAcronyms(acronyms, s), p, mapEntityAcronyms(acronyms, o), sentence) for (s,p,o) in dygiepp[sentence]['relations']]

	for sentence in llm:
		ok_relations_llm += [(mapEntityAcronyms(acronyms, s), p, mapEntityAcronyms(acronyms, o), sentence) for (s,p,o) in llm[sentence]['relations']]


	return list(ok_entities), list(set(ok_relations_dygiepp)), list(set(ok_relations_llm))


def extraction(filename):
	if filename[-5:] != '.json':
		return

	print('> processing: ' + filename)
	fw = open(output_dir + filename, 'w+')

	print('> processing: ' + filename + ' metadata reading')
	f = open(dataset_dump_dir + filename, 'r')
	paper2metadata = {}
	for row in f:
		drow = json.loads(row)
		paper_id = drow['_id']
		paper2metadata[paper_id] = drow['_source']
	f.close()

	print('> processing: ' + filename + ' dygiepp reading')
	f = open(dygiepp_output_dump_dir + filename, 'r')
	paper2dygiepp = {}
	for row in f:
		drow = json.loads(row)
		paper2dygiepp[drow['doc_key']] = getDygieppResults(drow)
	f.close()

	print('> processing: ' + filename + ' Llm annotation')
	f = open(llm_output_dump_dir + filename, 'r')
	paper2llm = {}
	for row in f:
		drow = json.loads(row)
		paper2llm[drow['doc_key']] = getLlmResults(drow)
	f.close()

	nlp = StanfordCoreNLP('http://localhost', port=9050)
	paper2openie = {}
	print('> processing: ' + filename + ' core nlp extraction')
	for index,paper_id in enumerate(tqdm(paper2metadata.keys(), total=len(paper2metadata.keys()), desc="Processing articles in the dataset:")):
		if paper_id in paper2dygiepp:
			corenlp_out = {}
			props = {'annotators': 'openie,tokenize,pos,depparse', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
			try:
				text_data = paper2metadata[paper_id]['title'].encode('utf8', 'ignore').decode('ascii', 'ignore') + '. ' + paper2metadata[paper_id]['abstract'].encode('utf8', 'ignore').decode('ascii', 'ignore')
				corenlp_out = json.loads(nlp.annotate(text_data, properties=props))
				openie_triples = getOpenieTriples(corenlp_out, paper2dygiepp[paper_id], paper2metadata[paper_id]['cso_semantic_topics'] +   paper2metadata[paper_id]['cso_syntactic_topics'])
				pos_triples = getPosTriples(corenlp_out,  paper2dygiepp[paper_id], paper2metadata[paper_id]['cso_semantic_topics'] +   paper2metadata[paper_id]['cso_syntactic_topics'])
				dependency_triples = getDependencyTriples(corenlp_out,  paper2dygiepp[paper_id], paper2metadata[paper_id]['cso_semantic_topics'] + paper2metadata[paper_id]['cso_syntactic_topics'])
				print('Here!')
				entities, dygiepp_triples, llm_triples = manageEntitiesAndDygieepRelations(paper2dygiepp[paper_id],paper2llm[paper_id], paper2metadata[paper_id]['cso_semantic_topics'] +   paper2metadata[paper_id]['cso_syntactic_topics'])

				data_input_for_dygepp = json.dump({
						'doc_key' : str(paper_id),
						'entities' : list(entities),
						'openie_triples' : list(openie_triples),
						'pos_triples' : list(pos_triples),
						'dependency_triples' : list(dependency_triples),
						'dygiepp_triples' : list(dygiepp_triples),
						'llm_triples': list(llm_triples)
					},fw)
				fw.write('\n')
			except Exception as e:
				print(e)
	fw.flush()
	fw.close()


def extraction(filename,booleanArgument):
	if filename[-5:] != '.json':
		return

	print('> processing: ' + filename)
	fw = open(output_dir + filename, 'w+', encoding="utf-8")
	fw1 = open(output_dir + Path(filename).stem + '_acronyms.txt', 'w+', encoding="utf-8")


	print('> processing: ' + filename + ' metadata reading')
	f = open(dataset_dump_dir + filename, 'r')
	paper2metadata = {}
	for row in f:
		drow = json.loads(row)
		paper_id = drow['_id']
		paper2metadata[paper_id] = drow['_source']
	f.close()

	print('> processing: ' + filename + ' dygiepp reading')
	f = open(dygiepp_output_dump_dir + filename, 'r')
	paper2dygiepp = {}
	for row in f:
		drow = json.loads(row)
		paper2dygiepp[drow['doc_key']] = getDygieppResults(drow)
	f.close()

	paper2llm = {}
	if booleanArgument:
		print('> processing: ' + filename + ' Llm annotation')
		f = open(llm_output_dump_dir + filename, 'r')
		for row in f:
			drow = json.loads(row)
			paper2llm[drow['doc_key']] = getLlmResults(drow)
		f.close()

	print('number of dygiepp papers : ' + str(len(paper2dygiepp)))
	print('number of llm papers : ' + str(len(paper2llm)))
	nlp = StanfordCoreNLP('http://localhost', port=9050)
	paper2openie = {}


	print('> processing: ' + filename + ' core nlp extraction')
	acronyms_global = {}
	for index,paper_id in enumerate(tqdm(paper2metadata.keys(), total=len(paper2metadata.keys()), desc="Processing articles in the dataset:")):
		if paper_id in paper2dygiepp and paper_id in paper2llm:
			corenlp_out = {}
			dygieppLlmAcronyms = {}
			openieAcronyms = {}
			posAcronyms = {}
			dependencyAcronyms = {}
			props = {'annotators': 'openie,tokenize,pos,depparse', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
			try:
				text_data = paper2metadata[paper_id]['title'].encode('utf8', 'ignore').decode('ascii', 'ignore') + '. ' + paper2metadata[paper_id]['abstract'].encode('utf8', 'ignore').decode('ascii', 'ignore')
				corenlp_out = json.loads(nlp.annotate(text_data, properties=props))

				dygiepp_entities = []
				llm_entities = []
				cso_topics = paper2metadata[paper_id]['cso_semantic_topics'] + paper2metadata[paper_id]['cso_syntactic_topics']
				for sentence in paper2dygiepp[paper_id]:
					dygiepp_entities += [x for (x, xtype) in paper2dygiepp[paper_id][sentence]['entities']]
				for sentence in paper2llm[paper_id]:
					llm_entities += [x for (x, xtype) in paper2llm[paper_id][sentence]['entities']]

				print('dygiepp_entities : ' + str(len(dygiepp_entities)))
				print('llm_entities : ' + str(len(llm_entities)))
				entities = dygiepp_entities + llm_entities + cso_topics
				acronyms = detectAcronymsLenient(entities)
				acronyms_global[paper_id] = acronyms
				openie_triples = getOpenieTriples(corenlp_out, paper2dygiepp[paper_id], paper2metadata[paper_id]['cso_semantic_topics'] +   paper2metadata[paper_id]['cso_syntactic_topics'], acronyms)
				pos_triples = getPosTriples(corenlp_out,  paper2dygiepp[paper_id], paper2metadata[paper_id]['cso_semantic_topics'] +   paper2metadata[paper_id]['cso_syntactic_topics'],acronyms )
				dependency_triples = getDependencyTriples(corenlp_out,  paper2dygiepp[paper_id], paper2metadata[paper_id]['cso_semantic_topics'] + paper2metadata[paper_id]['cso_syntactic_topics'], acronyms)
				entities, dygiepp_triples, llm_triples = manageEntitiesAndDygieepRelations(paper2dygiepp[paper_id],paper2llm[paper_id], entities, acronyms)

				json.dump({
						'doc_key' : str(paper_id),
						'entities' : list(entities),
						'openie_triples' : list(openie_triples),
						'pos_triples' : list(pos_triples),
						'dependency_triples' : list(dependency_triples),
						'dygiepp_triples' : list(dygiepp_triples),
						'llm_triples': list(llm_triples)
					},fw)
				fw.write('\n')
				fw1.write(paper_id + ':' + str(acronyms))
				fw1.write('\n')
			except Exception as e:
				print(e)
	merged_acronyms = {}  # Initialize an empty dictionary of global-level acronym mapping
	conflict_count = 0
	conflict_acronyms = {}
	key_values = defaultdict(set)
	for d in list(acronyms_global.values()):  # Iterate over all dictionaries in the list
		for key, value in d.items():
			key_values[key].add(value)
		merged_acronyms.update(d)  # Update with the latest dictionary's values
	for key, values in key_values.items():
		if len(values) > 1:
			conflict_count += 1
			conflict_acronyms[key] = values
	print('Number of conflicting acronyms: ' + str(conflict_count))
	with open(acro_output_dir + Path(filename).stem + '_global_acronyms.json', 'w', encoding="utf-8") as fw2:
		json.dump(merged_acronyms, fw2, indent=4, ensure_ascii=False)
	with open(debug_output_dir + Path(filename).stem + '_conflicting_acronyms.json', 'w', encoding="utf-8") as fw2:
		json.dump(merged_acronyms, fw2, indent=4, ensure_ascii=False)


	fw.flush()
	fw.close()
	fw1.flush()
	fw1.close()

if __name__ == '__main__':
	print(len(sys.argv))
	if len(sys.argv) == 3:
		importLLMData = sys.argv[2]

	else:
		print('> python corenlp_extractor.py importLLMout')
		exit(1)
	# Use functools.partial to fix the second argument

	print(importLLMData)
	boolean_value = importLLMData == "True"
	extraction_with_bool = partial(extraction, booleanArgument=boolean_value)


	files_to_parse = [filename for filename in os.listdir(dataset_dump_dir)]
	print(len(files_to_parse))
	pool = mp.Pool(10)
	result = pool.map(extraction_with_bool, files_to_parse)




	#detectAcronyms([('machine learning (ML)', 'A'), ('danilo dessi', 'B'), ('natural language processing (NLP)', 'C'), ('Natural Language Processing (NLP)', 'C')])


	

	 








