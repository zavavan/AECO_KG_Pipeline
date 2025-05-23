from collections import defaultdict

from EntitiesValidator import EntitiesValidator
from RelationsManager import RelationsManager
from EntitiesCleaner import EntitiesCleaner
from EntitiesMapper import EntitiesMapper
from KGDataDumper import KGDataDumper
import pickle
import json
import os
import gc



class TriplesGenerator:
	def __init__(self):
		self.entities2files = {}
		self.dygiepp2files = {}
		self.llm2files = {}
		self.openie2files = {}
		self.pos2files = {}
		self.dependency2files = {}
		self.data_extracted_dir = '../../outputs/extracted_triples/'
		self.acro_output_dir = '../../outputs/extracted_triples/acronyms/'
		self.debug_output_dir = '../../outputs/extracted_triples/debug/'
		self.global_acronyms = {}
		self.e2selected_type = {}
		self.e2openalex = {}
		self.e2cso = {} 
		self.e2dbpedia = {}
		self.e2wikidata = {}

		if not os.path.exists(self.debug_output_dir):
			os.makedirs(self.debug_output_dir)
			print(f"Folder created: {self.debug_output_dir}")
		else:
			print(f"Folder already exists: {self.debug_output_dir}")

	############ Data Loading #######################################################################################################

	def addDataInTripleDict(self, dic, triples_list, doc_key):
		for (s,p,o) in triples_list:
			if (s,p,o) not in dic:
				dic[(s,p,o)] = []
			dic[(s,p,o)] += [doc_key]

	# updated method to map triple to pairs (doc_key, sent_index) instead of to doc_key only
	def addDataInTripleDictSentences(self, dic, triples_list, doc_key):
		for (s, p, o, sent_index) in triples_list:
			if (s, p, o) not in dic:
				dic[(s, p, o)] = []
			dic[(s, p, o)].append((doc_key, sent_index))

	def loadData(self):
		for filename in os.listdir(self.data_extracted_dir):
			#c = 0
			if filename[-5:] == '.json':
				f = open(self.data_extracted_dir + filename, 'r').readlines()
				for row in f:
					try:
						paper_data = json.loads(row.strip())
						self.addDataInTripleDictSentences(self.dygiepp2files, paper_data['dygiepp_triples'], paper_data['doc_key'])
						self.addDataInTripleDictSentences(self.openie2files, paper_data['openie_triples'], paper_data['doc_key'])
						self.addDataInTripleDictSentences(self.pos2files, paper_data['pos_triples'], paper_data['doc_key'])
						self.addDataInTripleDictSentences(self.dependency2files, paper_data['dependency_triples'], paper_data['doc_key'])
						self.addDataInTripleDictSentences(self.llm2files, paper_data['llm_triples'],paper_data['doc_key'])
						for (e, etype) in paper_data['entities']:
							if (e, etype) not in self.entities2files:
								self.entities2files[(e, etype)] = []
							self.entities2files[(e, etype)] += [paper_data['doc_key']]
					except:
						pass


	###################################################################################################################################

	########### CLeaning of entities ##################################################################################################

	def applyCleanerMap(self, relations2files, cleaner_map):
		tool_triples2files = {}
		for (s,p,o),(files_sents) in relations2files.items():
			if s in cleaner_map and o in cleaner_map:
				if (cleaner_map[s],p,cleaner_map[o]) in tool_triples2files:
					tool_triples2files[(cleaner_map[s],p,cleaner_map[o])].update(set(files_sents))
				else:
					tool_triples2files[(cleaner_map[s],p,cleaner_map[o])] = set(files_sents)
		return tool_triples2files


	def updateThroughCleanerMap(self, cleaner_map):
		tmp_entities2files = {}
		for (e, e_type),files in self.entities2files.items():
			if e in cleaner_map:
				if (cleaner_map[e], e_type) in tmp_entities2files:
					tmp_entities2files[(cleaner_map[e], e_type)].update(set(files))
				else:
					tmp_entities2files[(cleaner_map[e], e_type)] = set(files)
		self.entities2files = tmp_entities2files

		self.dygiepp2files = self.applyCleanerMap(self.dygiepp2files, cleaner_map)
		self.llm2files = self.applyCleanerMap(self.llm2files, cleaner_map)
		self.pos2files = self.applyCleanerMap(self.pos2files, cleaner_map)
		self.openie2files = self.applyCleanerMap(self.openie2files, cleaner_map)
		self.dependency2files = self.applyCleanerMap(self.dependency2files, cleaner_map)

	###################################################################################################################################

	########### Mapping Entities to Global-Level Acronyms ##################################################################################################

	def load_global_acronyms(self):
		mappings_list = []
		for filename in os.listdir(self.acro_output_dir):
			if filename.endswith(".json"):  # Process only JSON files
				file_path = os.path.join(self.acro_output_dir, filename)
				try:
					with open(file_path, "r", encoding="utf-8") as file:
						data = json.load(file)
						mappings_list.append(data)
				except Exception as e:
					print(f"Error reading {filename}: {e}")
		merged_acronyms = {}  # Initialize an empty dictionary of global-level acronym mapping
		key_values = defaultdict(set)
		for d in mappings_list:
			for key, value in d.items():
				key_values[key].add(value)

		for k, v in key_values.items():
			# if acronyms as conflicting mappings, chose the longest one (e.g. "nzeb":["- zero energy building", "net - zero energy building"] --> "nzeb":"net - zero energy building")
			if len(v) > 1:
				merged_acronyms[k] = max(list(v), key=len)
			# otherwise map to the only available elemennt in the set
			else:
				merged_acronyms[k] = next(iter(v), None)

		self.global_acronyms = merged_acronyms



	def applyAcronymMap(self, relations2files, global_acronyms):
		tool_triples2files = {}
		for (s, p, o), (files_sents) in relations2files.items():
			if s.lower() in global_acronyms and o.lower() in global_acronyms:
				if (global_acronyms[s.lower()], p, global_acronyms[o.lower()]) in tool_triples2files:
					tool_triples2files[(global_acronyms[s.lower()], p, global_acronyms[o.lower()])].update(set(files_sents))
				else:
					tool_triples2files[(global_acronyms[s.lower()], p, global_acronyms[o.lower()])] = set(files_sents)
			elif s.lower() in global_acronyms:
				if (global_acronyms[s.lower()], p, o) in tool_triples2files:
					tool_triples2files[(global_acronyms[s.lower()], p, o)].update(set(files_sents))
				else:
					tool_triples2files[(global_acronyms[s.lower()], p, o)] = set(files_sents)
			elif o.lower() in global_acronyms:
				if (s, p, global_acronyms[o.lower()]) in tool_triples2files:
					tool_triples2files[(s, p, global_acronyms[o.lower()])].update(set(files_sents))
				else:
					tool_triples2files[(s, p, global_acronyms[o.lower()])] = set(files_sents)
			else:
				tool_triples2files[(s, p, o)] = set(files_sents)

		return tool_triples2files

	def updateThroughAcronymMap(self, global_acronyms):
		with open(self.debug_output_dir + 'entity_acronym_mapping.txt', 'w', encoding="utf-8") as fw:
			tmp_entities2files = {}
			mapped_counter=0
			for (e, e_type), fileSents in self.entities2files.items():
				if e.lower() in global_acronyms:
					mapped_counter+=1
					fw.write(e + ' : ' + global_acronyms[e.lower()] + '\n')
					if (e.lower(), e_type) in tmp_entities2files:
						tmp_entities2files[(global_acronyms[e.lower()], e_type)].update(set(fileSents))
					else:
						tmp_entities2files[(global_acronyms[e.lower()], e_type)] = set(fileSents)

				else:
					tmp_entities2files[(e, e_type)] = set(fileSents)

			self.entities2files = tmp_entities2files

		print('Number of acronym mapped entities: ' + str(mapped_counter) + ' out of ' + str(len(self.entities2files)))
		self.dygiepp2files = self.applyAcronymMap(self.dygiepp2files, global_acronyms)
		self.llm2files = self.applyAcronymMap(self.llm2files, global_acronyms)
		self.pos2files = self.applyAcronymMap(self.pos2files, global_acronyms)
		self.openie2files = self.applyAcronymMap(self.openie2files, global_acronyms)
		self.dependency2files = self.applyAcronymMap(self.dependency2files, global_acronyms)


	###################################################################################################################################



	############ Validation of entities ###############################################################################################

	def applyValidEntities(self, validEntities, relations2files):
		new_relations2files = {}
		for (s,p,o),file_sents in relations2files.items():
			if s in validEntities and o in validEntities:
				if (s,p,o) in new_relations2files:
					new_relations2files[(s,p,o)].update(set(file_sents))
				else:
					new_relations2files[(s,p,o)] = set(file_sents)
		return new_relations2files
	
	def updateThroughValidEntities(self, validEntities):

		tmp_entities2files = {}
		for (e, e_type),files in self.entities2files.items():
			if e in validEntities:
				if (e, e_type) in tmp_entities2files:
					tmp_entities2files[(e, e_type)].update(set(files))
				else:
					tmp_entities2files[(e, e_type)] = set(files)
		self.entities2files = tmp_entities2files

		self.dygiepp2files = self.applyValidEntities(validEntities, self.dygiepp2files)
		self.llm2files = self.applyValidEntities(validEntities, self.llm2files)
		self.openie2files = self.applyValidEntities(validEntities, self.openie2files)
		self.pos2files = self.applyValidEntities(validEntities, self.pos2files)
		self.dependency2files = self.applyValidEntities(validEntities, self.dependency2files)

	###################################################################################################################################
	

	########################################### Entities type and frequencies ##########################################################
	def entitiesTyping(self):
		self.e2types = {}
		for (e, e_type), files in self.entities2files.items():
			if e not in self.e2types:
				self.e2types[e] = {}
			
			if e_type != 'Generic':
				self.e2types[e][e_type] = len(files)
			else:
				if 'OtherScientificTerm' in self.e2types[e]:
					self.e2types[e]['OtherScientificTerm'] += len(files)
				else:
					self.e2types[e]['OtherScientificTerm'] = len(files)

		for e in self.e2types:
			occurence_count = self.e2types[e]
			
			# most frequent ignoring OtherEntity and CSOTopic
			selected_type = None
			max_freq = 0
			for etype,freq in dict(occurence_count).items():
				if etype != 'OtherScientificTerm' and etype != 'CSO Topic' and freq > max_freq:
					selected_type = etype
					max_freq = freq

			# if no Material, Method, etc. OtherEntity
			if selected_type == None:
				selected_type = 'OtherScientificTerm'

			self.e2selected_type[e] = selected_type
		
		with open('../../resources/e2selected_type.pickle', 'wb') as f:
			pickle.dump(self.e2selected_type, f)


	def entitiesFreq(self, cut_freq):
		e2count = {}
		for data_dict in [self.dygiepp_pair2info, self.llm_pair2info,  self.openie_pair2info, self.pos_pair2info, self.dep_pair2info]:
			for (s,o) in data_dict:
				if s not in e2count: e2count[s] = 0
				if o not in e2count: e2count[o] = 0
				e2count[s] += len(data_dict[(s,o)])
				e2count[o] += len(data_dict[(s,o)])
		
		return [e for e,c in e2count.items() if c >= cut_freq]



	###################################################################################################################################



	def createCheckpoint(self, name, els):
		with open('./ckpts/' + name + '.pickle', 'wb') as f:
			pickle.dump(els, f)

	def loadCheckpoint(self, name):
		with open('./ckpts/' + name + '.pickle', 'rb') as f:		
			return pickle.load(f)

	def run(self):


		ckpts_loading = os.path.exists('./ckpts/loading.pickle')
		ckpts_cleaning = os.path.exists('./ckpts/cleaning.pickle')
		ckpts_validation = os.path.exists('./ckpts/validation.pickle')
		ckpts_mapping = os.path.exists('./ckpts/mapping.pickle')
		ckpts_relations_handler= os.path.exists('./ckpts/relations_handler.pickle')


		print('--------------------------------------')
		print('>> Loading')
		if ckpts_loading and not ckpts_cleaning and not ckpts_validation and not ckpts_relations_handler and not ckpts_mapping:
			print('\t>> Loaded from ckpts')
			self.dygiepp2files, self.llm2files, self.openie2files, self.pos2files, self.dependency2files, self.entities2files = self.loadCheckpoint('loading')
			print(' \t- dygiepp triples:\t', len(self.dygiepp2files))
			print(' \t- llm triples:\t', len(self.llm2files))
			print(' \t- openie triples:\t', len(self.openie2files))
			print(' \t- pos triples:\t\t', len(self.pos2files))
			print(' \t- dep triples:\t\t', len(self.dependency2files))
		elif not ckpts_loading and not ckpts_cleaning and not ckpts_validation and not ckpts_relations_handler and not ckpts_mapping:
			self.loadData()
			self.load_global_acronyms()
			self.updateThroughAcronymMap(self.global_acronyms)
			self.createCheckpoint('loading', (self.dygiepp2files, self.llm2files, self.openie2files, self.pos2files, self.dependency2files, self.entities2files))
			print(' \t- dygiepp triples:\t', len(self.dygiepp2files))
			print(' \t- llm triples:\t', len(self.llm2files))
			print(' \t- openie triples:\t', len(self.openie2files))
			print(' \t- pos triples:\t\t', len(self.pos2files))
			print(' \t- dep triples:\t\t', len(self.dependency2files))
		else:
			print('\t>> skipped')
		print('--------------------------------------')
		print('>> Entity cleaning')
		if ckpts_cleaning and not ckpts_validation and not ckpts_relations_handler and not ckpts_mapping:
			print('\t>> Loaded from ckpts')
			self.dygiepp2files, self.llm2files, self.openie2files, self.pos2files, self.dependency2files, self.entities2files = self.loadCheckpoint('cleaning')
			print(' \t- dygiepp triples:\t', len(self.dygiepp2files))
			print(' \t- llm triples:\t', len(self.llm2files))
			print(' \t- openie triples:\t', len(self.openie2files))
			print(' \t- pos triples:\t\t', len(self.pos2files))
			print(' \t- dep triples:\t\t', len(self.dependency2files))
		elif not ckpts_cleaning and not ckpts_validation and not ckpts_relations_handler and not ckpts_mapping:
			ec = EntitiesCleaner(set([e for (e,e_type) in self.entities2files.keys()]))
			ec.run()
			cleaner_map = ec.get()
			self.updateThroughCleanerMap(cleaner_map)
			del cleaner_map
			gc.collect()
			self.createCheckpoint('cleaning', (self.dygiepp2files, self.llm2files, self.openie2files, self.pos2files, self.dependency2files, self.entities2files))
			print(' \t- dygiepp triples:\t', len(self.dygiepp2files))
			print(' \t- llm triples:\t', len(self.llm2files))
			print(' \t- openie triples:\t', len(self.openie2files))
			print(' \t- pos triples:\t\t', len(self.pos2files))
			print(' \t- dep triples:\t\t', len(self.dependency2files))
		else:
			print('\t>> skipped')
		print('--------------------------------------')

		print('>> Entity validation')
		if ckpts_validation and not ckpts_relations_handler and not ckpts_mapping:
			print('\t>> Loaded from ckpts')
			self.dygiepp2files, self.llm2files, self.openie2files, self.pos2files, self.dependency2files, self.entities2files = self.loadCheckpoint('validation')
			print(' \t- dygiepp triples:\t', len(self.dygiepp2files))
			print(' \t- llm triples:\t', len(self.llm2files))
			print(' \t- openie triples:\t', len(self.openie2files))
			print(' \t- pos triples:\t\t', len(self.pos2files))
			print(' \t- dep triples:\t\t', len(self.dependency2files))
		elif not ckpts_validation and not ckpts_relations_handler and not ckpts_mapping:
			ev = EntitiesValidator(self.entities2files)
			ev.run()
			valid_entities = ev.get()

			self.updateThroughValidEntities(valid_entities)
			del ev
			gc.collect()
			self.createCheckpoint('validation', (self.dygiepp2files, self.llm2files, self.openie2files, self.pos2files, self.dependency2files, self.entities2files))
			print(' \t- dygiepp triples:\t', len(self.dygiepp2files))
			print(' \t- llm triples:\t', len(self.llm2files))
			print(' \t- openie triples:\t', len(self.openie2files))
			print(' \t- pos triples:\t\t', len(self.pos2files))
			print(' \t- dep triples:\t\t', len(self.dependency2files))
		else:
			print('\t>> skipped')
		print('--------------------------------------')


		print('>> Relations handling')
		if ckpts_relations_handler:
			print('\t>> Loaded from ckpts')
			self.dygiepp_pair2info, self.llm_pair2info, self.openie_pair2info, self.pos_pair2info, self.dep_pair2info, self.entities2files = self.loadCheckpoint('relations_handler')
			print(' \t- dygiepp pairs:\t', len(self.dygiepp_pair2info))
			print(' \t- llm pairs:\t', len(self.llm_pair2info))
			print(' \t- openie pairs:\t\t', len(self.openie_pair2info))
			print(' \t- pos pairs:\t\t', len(self.pos_pair2info))
			print(' \t- dep pairs:\t\t', len(self.dep_pair2info))
		elif not ckpts_relations_handler and not ckpts_mapping:
			rm = RelationsManager(self.dygiepp2files, self.llm2files, self.pos2files, self.openie2files, self.dependency2files)
			rm.run()
			self.dygiepp_pair2info, self.llm_pair2info, self.pos_pair2info, self.openie_pair2info, self.dep_pair2info = rm.get()
			del rm
			del self.dygiepp2files
			del self.llm2files
			del self.openie2files
			del self.pos2files
			del self.dependency2files
			gc.collect()
			self.createCheckpoint('relations_handler', (self.dygiepp_pair2info, self.llm_pair2info, self.openie_pair2info, self.pos_pair2info, self.dep_pair2info, self.entities2files))
			print(' \t- dygiepp pairs:\t', len(self.dygiepp_pair2info))
			#print(self.dygiepp_pair2info)
			print(' \t- llm pairs:\t', len(self.llm_pair2info))
			print(' \t- openie pairs:\t\t', len(self.openie_pair2info))
			print(' \t- pos pairs:\t\t', len(self.pos_pair2info))
			print(' \t- dep pairs:\t\t', len(self.dep_pair2info))
		else:
			print('\t>> skipped')
		print('--------------------------------------')

		print('>> Mapping to external resources')
		if ckpts_mapping:
			print('\t>> Loaded from ckpts')
			self.e2openalex, self.e2cso, self.e2dbpedia, self.e2wikidata= self.loadCheckpoint('mapping')
		elif not ckpts_mapping:
			all_pairs = set(self.dygiepp_pair2info.keys()) | set(self.llm_pair2info.keys()) | set(self.pos_pair2info.keys()) | set(self.openie_pair2info.keys()) | set(self.dep_pair2info.keys())
			#mapper = EntitiesMapper([e for e, t in self.entities2files.keys()], all_pairs)
			cut_freq = 1
			mapper = EntitiesMapper(self.entitiesFreq(cut_freq),self.entities2files, all_pairs)
			mapper.run()
			self.e2openalex, self.e2cso, self.e2dbpedia, self.e2wikidata = mapper.getMaps()
			del mapper
			gc.collect()
			self.createCheckpoint('mapping', (self.e2openalex, self.e2cso, self.e2dbpedia, self.e2wikidata))
		else:
			print('\t>> skipped')
		print('--------------------------------------')
		print('>> Data dumping and merging')
		self.entitiesTyping()
		dumper = KGDataDumper(self.dygiepp_pair2info, self.llm_pair2info,  self.pos_pair2info, self.openie_pair2info, self.dep_pair2info, self.e2openalex, self.e2cso, self.e2dbpedia, self.e2wikidata, self.e2selected_type)
		dumper.run()
		print('--------------------------------------')



if __name__ == '__main__':
	if not os.path.exists('./ckpts/'):
		os.makedirs('./ckpts/')
	tg = TriplesGenerator()
	tg.run()
	


