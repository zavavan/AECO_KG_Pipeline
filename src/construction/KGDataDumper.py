from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from sentence_transformers import util
import pandas as pd
import numpy as np
import pickle
import time
import csv 
import os
import json

class KGDataDumper:
	def __init__(self, dygiepp_pair2info, llm_pair2info, pos_pair2info, openie_pair2info, dep_pair2info, e2openalex, e2cso, e2dbpedia, e2wikidata, e2type):

		self.dygiepp_pair2info = dygiepp_pair2info
		self.llm_pair2info = llm_pair2info
		self.pos_pair2info = pos_pair2info
		self.openie_pair2info = openie_pair2info
		self.dep_pair2info = dep_pair2info
		self.e2openalex = e2openalex
		self.e2cso = e2cso
		self.e2dbpedia = e2dbpedia
		self.e2wikidata = e2wikidata
		self.e2type = e2type

		self.pair2info = {}
		self.cskg2cso = {}
		self.cskg2wikidata = {}
		self.cskg2dbpedia = {}
		self.validDomainRelRange = set()
		self.label2cskg_entity = {}
		self.triples = []

		self.triples_csv_filename = './cskg_data/cskg_triples.csv'
		self.debug_output_dir = '../../outputs/extracted_triples/debug/'

	def collectInfo_w_sentIndexes(self):
		pairs = set(self.dygiepp_pair2info.keys()) | set(self.llm_pair2info.keys()) | set(
			self.pos_pair2info.keys()) | set(self.openie_pair2info.keys()) | set(self.dep_pair2info.keys())

		for (s, o) in pairs:
			if (s, o) not in self.pair2info:
				self.pair2info[(s, o)] = {}

			if (s, o) in self.dygiepp_pair2info.keys():
				for rel in self.dygiepp_pair2info[(s, o)]:
					if rel not in self.pair2info[(s, o)]:
						self.pair2info[(s, o)][rel] = {'file_sents': list(self.dygiepp_pair2info[(s, o)][rel])}
						self.pair2info[(s, o)][rel]['source'] = ['dygiepp']
					else:
						self.pair2info[(s, o)][rel]['file_sents'] += list(self.dygiepp_pair2info[(s, o)][rel])
						self.pair2info[(s, o)][rel]['source'] += ['dygiepp']

			if (s, o) in self.llm_pair2info.keys():
				for rel in self.llm_pair2info[(s, o)]:
					if rel not in self.pair2info[(s, o)]:
						self.pair2info[(s, o)][rel] = {'file_sents': list(self.llm_pair2info[(s, o)][rel])}
						self.pair2info[(s, o)][rel]['source'] = ['llm']
					else:
						self.pair2info[(s, o)][rel]['file_sents'] += list(self.llm_pair2info[(s, o)][rel])
						self.pair2info[(s, o)][rel]['source'] += ['llm']

			if (s, o) in self.pos_pair2info.keys():
				for rel in self.pos_pair2info[(s, o)]:
					if rel not in self.pair2info[(s, o)]:
						self.pair2info[(s, o)][rel] = {'file_sents': list(self.pos_pair2info[(s, o)][rel])}
						self.pair2info[(s, o)][rel]['source'] = ['pos tagger']
					else:
						self.pair2info[(s, o)][rel]['file_sents'] += list(self.pos_pair2info[(s, o)][rel])
						self.pair2info[(s, o)][rel]['source'] += ['pos tagger']

			if (s, o) in self.openie_pair2info.keys():
				for rel in self.openie_pair2info[(s, o)]:
					if rel not in self.pair2info[(s, o)]:
						self.pair2info[(s, o)][rel] = {'file_sents': list(self.openie_pair2info[(s, o)][rel])}
						self.pair2info[(s, o)][rel]['source'] = ['openie']
					else:
						self.pair2info[(s, o)][rel]['file_sents'] += list(self.openie_pair2info[(s, o)][rel])
						self.pair2info[(s, o)][rel]['source'] += ['openie']

			if (s, o) in self.dep_pair2info.keys():
				for rel in self.dep_pair2info[(s, o)]:
					if rel not in self.pair2info[(s, o)]:
						self.pair2info[(s, o)][rel] = {'file_sents': list(self.dep_pair2info[(s, o)][rel])}
						self.pair2info[(s, o)][rel]['source'] = ['dependency tagger']
					else:
						self.pair2info[(s, o)][rel]['file_sents'] += list(self.dep_pair2info[(s, o)][rel])
						self.pair2info[(s, o)][rel]['source'] += ['dependency tagger']


	def collectInfo(self):
		pairs = set(self.dygiepp_pair2info.keys()) | set(self.llm_pair2info.keys()) | set(self.pos_pair2info.keys()) | set(self.openie_pair2info.keys()) | set(self.dep_pair2info.keys())
		
		for (s,o) in pairs:
			if (s,o) not in self.pair2info:
				self.pair2info[(s,o)] = {}
			
			if (s,o) in self.dygiepp_pair2info.keys():
				for rel in self.dygiepp_pair2info[(s,o)]:
					if rel not in self.pair2info[(s,o)]:
						self.pair2info[(s,o)][rel] = {'file' : list(self.dygiepp_pair2info[(s,o)][rel])}
						self.pair2info[(s,o)][rel]['source'] = ['dygiepp']				
					else:
						self.pair2info[(s,o)][rel]['files'] += list(self.dygiepp_pair2info[(s,o)][rel])
						self.pair2info[(s,o)][rel]['source'] += ['dygiepp']


			if (s,o) in self.llm_pair2info.keys():
				for rel in self.llm_pair2info[(s,o)]:
					if rel not in self.pair2info[(s,o)]:
						self.pair2info[(s,o)][rel] = {'files' : list(self.llm_pair2info[(s,o)][rel])}
						self.pair2info[(s,o)][rel]['source'] = ['llm']
					else:
						self.pair2info[(s,o)][rel]['files'] += list(self.llm_pair2info[(s,o)][rel])
						self.pair2info[(s,o)][rel]['source'] += ['llm']

			if (s,o) in self.pos_pair2info.keys():
				for rel in self.pos_pair2info[(s,o)]:
					if rel not in self.pair2info[(s,o)]:
						self.pair2info[(s,o)][rel] = {'files' : list(self.pos_pair2info[(s,o)][rel])}
						self.pair2info[(s,o)][rel]['source'] = ['pos tagger']				
					else:
						self.pair2info[(s,o)][rel]['files'] += list(self.pos_pair2info[(s,o)][rel])
						self.pair2info[(s,o)][rel]['source'] += ['pos tagger']


			if (s,o) in self.openie_pair2info.keys():
				for rel in self.openie_pair2info[(s,o)]:
					if rel not in self.pair2info[(s,o)]:
						self.pair2info[(s,o)][rel] = {'files' : list(self.openie_pair2info[(s,o)][rel])}
						self.pair2info[(s,o)][rel]['source'] = ['openie']				
					else:
						self.pair2info[(s,o)][rel]['files'] += list(self.openie_pair2info[(s,o)][rel])
						self.pair2info[(s,o)][rel]['source'] += ['openie']

			if (s,o) in self.dep_pair2info.keys():
				for rel in self.dep_pair2info[(s,o)]:
					if rel not in self.pair2info[(s,o)]:
						self.pair2info[(s,o)][rel] = {'files' : list(self.dep_pair2info[(s,o)][rel])}
						self.pair2info[(s,o)][rel]['source'] = ['dependency tagger']				
					else:
						self.pair2info[(s,o)][rel]['files'] += list(self.dep_pair2info[(s,o)][rel])
						self.pair2info[(s,o)][rel]['source'] += ['dependency tagger']


## This function harmonizes entity mappings from different knowledge sources (OpenAlex, CSO, DBpedia, Wikidata)
# into a single unified knowledge graph (CSKG). It ensures consistent entity naming (using the longest label), consolidation of duplicate entities from different sources
	def mergeEntities(self):

		openalex2cskg = {}
		cso2cskg = {}
		wikidata2cskg = {}
		dbpedia2cskg = {}

		print('size of external resource mappings')
		print("e2openalex: " + str(len(self.e2openalex)))
		print("e2cso: " + str(len(self.e2cso)))
		print("e2dbpedia: " + str(len(self.e2dbpedia)))
		print("e2wikidata: " + str(len(self.e2wikidata)))

		for (s,o) in self.pair2info:

			if s in self.e2openalex:
				if self.e2openalex[s] not in openalex2cskg:
					openalex2cskg[self.e2openalex[s]] = []
				openalex2cskg[self.e2openalex[s]] += [s]
			if s in self.e2cso:
				if self.e2cso[s] not in cso2cskg:
					cso2cskg[self.e2cso[s]] = []
				cso2cskg[self.e2cso[s]] += [s]
			if s in self.e2dbpedia:
				if self.e2dbpedia[s] not in dbpedia2cskg:
					dbpedia2cskg[self.e2dbpedia[s]] = []
				dbpedia2cskg[self.e2dbpedia[s]] += [s]
			if s in self.e2wikidata:
				if self.e2wikidata[s] not in wikidata2cskg:
					wikidata2cskg[self.e2wikidata[s]] = []
				wikidata2cskg[self.e2wikidata[s]] += [s]

			if o in self.e2openalex:
				if self.e2openalex[o] not in openalex2cskg:
					openalex2cskg[self.e2openalex[o]] = []
				openalex2cskg[self.e2openalex[o]] += [o]
			if o in self.e2cso:
				if self.e2cso[o] not in cso2cskg:
					cso2cskg[self.e2cso[o]] = []
				cso2cskg[self.e2cso[o]] += [o]
			if o in self.e2dbpedia:
				if self.e2dbpedia[o] not in dbpedia2cskg:
					dbpedia2cskg[self.e2dbpedia[o]] = []
				dbpedia2cskg[self.e2dbpedia[o]] += [o]
			if o in self.e2wikidata:
				if self.e2wikidata[o] not in wikidata2cskg:
					wikidata2cskg[self.e2wikidata[o]] = []
				wikidata2cskg[self.e2wikidata[o]] += [o]

		# merging with openalex
		for oae, oakg_entities_labels in openalex2cskg.items():
			oekg_entity = max(list(set(oakg_entities_labels)), key=len)  # longest label

			for label in list(set(oakg_entities_labels)):
				self.label2cskg_entity[label] = oekg_entity
			self.cskg2cso[oekg_entity] = oae

		# merging with cso
		for csoe, cskg_entities_labels in cso2cskg.items():
			cskg_entity = max(list(set(cskg_entities_labels)), key=len) #longest label
			
			for label in list(set(cskg_entities_labels)):
				self.label2cskg_entity[label] = cskg_entity
			self.cskg2cso[cskg_entity] = csoe


		# merging with dbpedia
		for dbe, cskg_entities_labels in dbpedia2cskg.items():
			
			# check if there exists an entity
			cskg_entity = None
			for label in list(set(cskg_entities_labels)):
				if label in self.label2cskg_entity:
					cskg_entity = self.label2cskg_entity[label]
					break

			if cskg_entity == None:
				cskg_entity = max(list(set(cskg_entities_labels)), key=len)
			
			for label in list(set(cskg_entities_labels)):
				self.label2cskg_entity[label] = cskg_entity
			self.cskg2dbpedia[cskg_entity] = dbe


		# merging with wikidata
		for wde, cskg_entities_labels in wikidata2cskg.items():
			
			# check if there exists an entity
			cskg_entity = None
			for label in list(set(cskg_entities_labels)):
				if label in self.label2cskg_entity:
					cskg_entity = self.label2cskg_entity[label]
					break

			if cskg_entity == None:
				cskg_entity = max(list(set(cskg_entities_labels)), key=len)

			for label in list(set(cskg_entities_labels)):
				self.label2cskg_entity[label] = cskg_entity
			self.cskg2wikidata[cskg_entity] = wde

		print('self.label2cskg_entity')
		print(len(self.label2cskg_entity))
		with open(os.path.join(self.debug_output_dir, 'label2cskg_entity_externalMapping.json'), 'w', encoding="utf-8") as fw1:
			json.dump(self.label2cskg_entity, fw1, indent=4, ensure_ascii=False)

	# function used by mergeEntitiesEuristic
	def mergeEntitiesEmbeddings(self, model, entities):

		paraphrases = util.paraphrase_mining(model, entities, query_chunk_size=100, corpus_chunk_size=10000, batch_size=256, top_k=5, show_progress_bar=False)

		for paraphrase in paraphrases:
			score, i, j = paraphrase
			ei = entities[i] # entity
			ej = entities[j] # entity 

			# since the results are ordered, the loop is stopped when the similarity is lower than 0.9
			if score < 0.70:
				break

			if ei not in self.label2cskg_entity and ej not in self.label2cskg_entity:
				self.label2cskg_entity[ej] = ei
				self.label2cskg_entity[ei] = ei
				#print(ej, '->', ei, ' : ', score)
			elif ei not in self.label2cskg_entity and ej in self.label2cskg_entity:
				self.label2cskg_entity[ei] = self.label2cskg_entity[ej]
				#print(ei, '->', ej, '->',  self.label2cskg_entity[ej], ' : ', score)
			elif ei in self.label2cskg_entity and ej not in self.label2cskg_entity:
				self.label2cskg_entity[ej] = self.label2cskg_entity[ei]
				#print(ej, '->', ei, '->',  self.label2cskg_entity[ei], ' : ', score)


	def mergeEntitiesEuristic(self):


		try:
			# merge lables with separate embeddings merging previously computed if it exists, otherwise it will be computed in the
			# execution flow of the code
			f = open('../../resources/only_embeddings_label2cskg_entity.pickle', 'rb')
			only_embeddings_label2cskg_entity = pickle.load(f)
			f.close()
			for (ei, ej) in only_embeddings_label2cskg_entity.items():	
				
				if ei not in self.label2cskg_entity and ej not in self.label2cskg_entity:
					self.label2cskg_entity[ej] = ei
					self.label2cskg_entity[ei] = ei
				elif ei not in self.label2cskg_entity and ej in self.label2cskg_entity:
					self.label2cskg_entity[ei] = self.label2cskg_entity[ej]
				elif ei in self.label2cskg_entity and ej not in self.label2cskg_entity:
					self.label2cskg_entity[ej] = self.label2cskg_entity[ei]
		except FileNotFoundError:

			# sentence-transformers/paraphrase-distilroberta-base-v2
			model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v2')
			word2entities = {}

			for (s,o) in self.pair2info:
				stokens = word_tokenize(s)
				otokens = word_tokenize(o)

				for t in stokens:
					if t not in word2entities:
						word2entities[t] = set()
					word2entities[t].add(s)

				for t in otokens:
					if t not in word2entities:
						word2entities[t] = set()
					word2entities[t].add(o)

			wordcount = len(word2entities)
			for word, entities in word2entities.items():
				#print(wordcount, word, len(entities))
				#wordcount -= 1
				if len(entities) > 1:
					self.mergeEntitiesEmbeddings(model, list(entities))
				#print('\t>> tokens to be checked:', wordcount)

		print('self.label2cskg_entity')
		with open(os.path.join(self.debug_output_dir,'label2cskg_entity_embeddingMapping.json'), 'w', encoding="utf-8") as fw2:
			json.dump(self.label2cskg_entity, fw2, indent=4, ensure_ascii=False)
		#print(self.label2cskg_entity)

	def createTriplesData(self):

		# triple creation starting from the existing relationships between pairs of entities. The merging of entities based on the approaches above is 
		# performed here.
		for (s,o) in self.pair2info:
			s_cskg = self.label2cskg_entity[s] if s in self.label2cskg_entity else s
			o_cskg = self.label2cskg_entity[o] if o in self.label2cskg_entity else o

			stype = 'OtherEntity'
			otype = 'OtherEntity'
			if s_cskg in self.e2type:
				stype = self.e2type[s_cskg].replace('OtherScientificTerm', 'OtherEntity')

			if o_cskg in self.e2type:
				otype = self.e2type[o_cskg].replace('OtherScientificTerm', 'OtherEntity')

			for rel in self.pair2info[(s,o)]:
				if s_cskg != o_cskg:
					self.triples += [(s_cskg, rel, o_cskg, len(set(self.pair2info[(s,o)][rel]['file_sents'])), self.pair2info[(s,o)][rel]['source'], self.pair2info[(s,o)][rel]['file_sents'], stype, otype)]


		# merging triples after entity mapping and merging
		triples2info = {}
		for (s, p, o, support, sources, fileSents, stype, otype) in self.triples:
			if (s,p,o) not in triples2info.keys():
				triples2info[(s,p,o)] = {
					'support' : len(fileSents),
					'sources' : set(sources),
					'fileSents' : set(fileSents),
					'subj_type' : stype,
					'obj_type' : otype,
					'source_len': len(sources) # add the number of suources based on the definition of support
				}
			else:
				new_sources = triples2info[(s,p,o)]['sources'] | set(sources)
				triples2info[(s,p,o)]['sources'] = new_sources

				new_fileSents = triples2info[(s,p,o)]['fileSents'] | set(fileSents)
				triples2info[(s,p,o)]['fileSents'] = new_fileSents

				triples2info[(s,p,o)]['source_len'] = len(new_sources)
				triples2info[(s,p,o)]['support'] = len(new_fileSents)

		#saving merged triples in dataframe
		subjs = []
		rels = []
		objs = []
		supports = []
		sources = []
		fileSents = []
		subj_types = []
		obj_types = []
		source_lens = []
		for (s,p,o) in triples2info.keys():
			subjs += [s]
			rels += [p]
			objs += [o]
			supports += [triples2info[(s,p,o)]['support']]
			sources += [triples2info[(s,p,o)]['sources']]
			fileSents += [triples2info[(s,p,o)]['fileSents']]
			subj_types += [triples2info[(s,p,o)]['subj_type']]
			obj_types += [triples2info[(s,p,o)]['obj_type']]
			source_lens += [triples2info[(s,p,o)]['source_len']]

		merged_triples = pd.DataFrame({'subj' : subjs, 'rel' : rels, 'obj' : objs, 'support' : supports, 'sources' : sources, 'fileSents' : fileSents, 'subj_type' : subj_types, 'obj_type' : obj_types, 'source_len': source_lens})
		merged_triples.sort_values(by=['support'], inplace=True)
		merged_triples.to_csv(self.triples_csv_filename, index=False)

		print('\t>> Number of triples:', len(triples2info.keys()))

		self.saveAsPickle(self.label2cskg_entity, 'label2cskg_entity')
		self.saveAsPickle(self.cskg2cso, 'cskg2cso')
		self.saveAsPickle(self.cskg2dbpedia, 'cskg2dbpedia')
		self.saveAsPickle(self.cskg2wikidata, 'cskg2wikidata')


	def saveAsPickle(self, data, objectName):
		pickle_out = open('./cskg_data/' + objectName + '.pickle', 'wb')
		pickle.dump(data, pickle_out)
		pickle_out.close()
		

	def run(self):
		if not os.path.exists('./cskg_data/'):
			os.makedirs('./cskg_data/')

		self.collectInfo_w_sentIndexes()
		self.mergeEntities()
		self.mergeEntitiesEuristic()
		self.createTriplesData()







