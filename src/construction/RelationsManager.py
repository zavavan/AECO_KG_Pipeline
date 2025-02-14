
from scipy import spatial
import pandas as pd
import numpy as np
import collections
import time
import csv

class RelationsManager:


	def __init__(self, dygieep_relations2file_sents, llm_relations2file_sents, stanfordcore_pos_relations2file_sents, stanfordcore_openie_relations2file_sents, stanfordcore_dep_relations2file_sents):
		self.dygieep_relations2file_sents = dygieep_relations2file_sents
		self.llm_relations2file_sents = llm_relations2file_sents
		self.stanfordcore_pos_relations2file_sents = stanfordcore_pos_relations2file_sents
		self.stanfordcore_openie_relations2file_sents = stanfordcore_openie_relations2file_sents
		self.stanfordcore_dep_relations2file_sents = stanfordcore_dep_relations2file_sents

		self.verb_map_path = '../../resources/CSKG_VerbNet_verb_map.csv'
		self.verb_map = {}

		self.dygiepp_pair2info = {}
		self.llm_pair2info = {}
		self.pos_pair2info = {}
		self.openie_pair2info = {}
		self.dep_pair2info = {}


	def loadVerbMap(self):
		verb_info = pd.read_csv(self.verb_map_path, sep=',')
		for i,r in verb_info.iterrows():
			for j in range(34):
				verb = r['v' + str(j)]
				if str(verb) != 'nan':
					self.verb_map[verb] = r['predicate']

	def bestLabelDygiepp(self):
		pairs = {}
		for (s,p,o), file_sents in self.dygieep_relations2file_sents.items():
			if (s,o) not in pairs:
				pairs[(s,o)] = {}
				pairs[(s,o)][p] = file_sents

		self.dygiepp_pair2info = pairs

	def bestLabelLlm(self):
		pairs = {}
		for (s,p,o), file_sents in self.llm_relations2file_sents.items():
			if (s,o) not in pairs:
				pairs[(s,o)] = {}
				pairs[(s,o)][p] = file_sents

		self.llm_pair2info = pairs


	def mapVerbRelations(self, verb_relations2file_sents):
		new_verb_relations2file_sents= {}

		for (s,p,o), file_sents in verb_relations2file_sents.items():
			if p in self.verb_map:
				mapped_verb = self.verb_map[p]

				if (s, mapped_verb, o) not in new_verb_relations2file_sents:
					new_verb_relations2file_sents[(s, mapped_verb, o)] = []
				new_verb_relations2file_sents[(s, mapped_verb, o)] += file_sents
				
		return new_verb_relations2file_sents


	def labelSelector(self, verb_relations2file_sents):
		pairs = {}
		for (s,p,o), file_sents in verb_relations2file_sents.items():
			if (s,o) not in pairs:
				pairs[(s,o)] = {}
			pairs[(s,o)][p] = file_sents

		return pairs

	

	def mapLlmRelations(self):
		llm_relations2file_sents = {}
		
		# the mapping of dygiepp is done accordingly to the mapping defined by AIKG_VerbNet_verb_map.csv 
		for (s,p,o), file_sents in self.llm_relations2file_sents.items():
			if p == 'USED-FOR':
				llm_relations2file_sents[(o, 'uses', s)] = file_sents
			elif p == 'FEATURE-OF' or p == 'PART-OF':
				llm_relations2file_sents[(o, 'includes', s)] = file_sents
			elif p == 'EVALUATE-FOR':
				llm_relations2file_sents[(s, 'analyzes', o)] = file_sents
			elif p == 'HYPONYM-OF':
				llm_relations2file_sents[(s, 'skos:broader/is/hyponym-of', o)] = file_sents
			elif p == 'COMPARE':
				llm_relations2file_sents[(s,'matches',o)] = file_sents

		self.llm_relations2file_sents = llm_relations2file_sents

	def mapDygieppRelations(self):
		dygieep_relations2file_sents = {}

		# the mapping of dygiepp is done accordingly to the mapping defined by AIKG_VerbNet_verb_map.csv
		for (s, p, o), file_sents in self.dygieep_relations2file_sents.items():
			if p == 'USED-FOR':
				dygieep_relations2file_sents[(o, 'uses', s)] = file_sents
			elif p == 'FEATURE-OF' or p == 'PART-OF':
				dygieep_relations2file_sents[(o, 'includes', s)] = file_sents
			elif p == 'EVALUATE-FOR':
				dygieep_relations2file_sents[(s, 'analyzes', o)] = file_sents
			elif p == 'HYPONYM-OF':
				dygieep_relations2file_sents[(s, 'skos:broader/is/hyponym-of', o)] = file_sents
			elif p == 'COMPARE':
				dygieep_relations2file_sents[(s, 'matches', o)] = file_sents

		self.dygieep_relations2file_sents = dygieep_relations2file_sents

	def run(self):
		start = time.time()
		self.loadVerbMap()

		self.stanfordcore_pos_relations2file_sents = self.mapVerbRelations(self.stanfordcore_pos_relations2file_sents)
		self.pos_pair2info = self.labelSelector(self.stanfordcore_pos_relations2file_sents)

		self.stanfordcore_openie_relations2file_sents = self.mapVerbRelations(self.stanfordcore_openie_relations2file_sents)
		self.openie_pair2info = self.labelSelector(self.stanfordcore_openie_relations2file_sents)

		self.stanfordcore_dep_relations2file_sents = self.mapVerbRelations(self.stanfordcore_dep_relations2file_sents)
		self.dep_pair2info = self.labelSelector(self.stanfordcore_dep_relations2file_sents)

		self.mapDygieppRelations()
		self.bestLabelDygiepp()
		self.mapLlmRelations()
		self.bestLabelLlm()
				
		

	def get(self):
		return self.dygiepp_pair2info, self.llm_pair2info, self.pos_pair2info, self.openie_pair2info, self.dep_pair2info
		













