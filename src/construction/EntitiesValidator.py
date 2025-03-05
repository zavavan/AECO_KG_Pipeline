from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from urllib.parse import unquote
from nltk.corpus import stopwords

import pickle
import json
import nltk
import csv
import os
import requests
from tqdm import tqdm
import time

class EntitiesValidator:
	
	def __init__(self, entities2files):
		self.entities2files = entities2files
		self.inputEntities = set([e for (e, e_type) in entities2files.keys()])
		self.csoResourcePath = '../../resources/CSO.3.1.csv'
		self.blacklist_path = '../../resources/blacklist.txt'
		self.mag_topics_dir = '../../dataset/computer_science/'
		self.open_alex_wikidata_concepts_path = '../../resources/wikidata_aeco.json'
		self.debug_output_dir = '../../outputs/extracted_triples/debug/'
		self.csoTopics = set()
		self.magTopics = set()
		self.wikidata_concepts = {}
		self.global_acronyms = {}
		self.validEntities = set()
		self.invalidEntities = set()
		self.blacklist = set()


	def loadCSOTopics(self):
		with open(self.csoResourcePath, 'r', encoding='utf-8') as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			for row in csv_reader:
				t1 = unquote(row[0]).replace('<https://', '')[:-1]
				t2 = unquote(row[2]).replace('<https://', '')[:-1]
				if t1.startswith('cso.kmi.open.ac.uk/topics/'):
					t1 = t1.split('/')[-1]
					self.csoTopics.add(t1.lower().replace('_',' '))
				if t2.startswith('cso.kmi.open.ac.uk/topics/'):
					t2 = t2.split('/')[-1]
					self.csoTopics.add(t2.lower().replace('_',' '))


	def loadBlacklist(self):
		with open(self.blacklist_path) as f:
			for line in f.readlines():
				self.blacklist.add(line.strip())

	def load_open_alex_concepts(self):
		# Open the file and read its content
		with open(self.open_alex_wikidata_concepts_path, 'r') as f:
			file_content = f.read()

		self.wikidata_concepts = json.loads(file_content)



	def loadMAGTopics(self):
		for filename in os.listdir(self.mag_topics_dir):
			topics = []
			if filename[-5:] == '.json':
				f = open(self.mag_topics_dir + filename, 'r')
				for row in f:
					paper_data = json.loads(row.strip())
					topics += paper_data['_source']['topics']
				f.close()
		self.magTopics = set(topics)



	def validation(self):
		swords = set(stopwords.words('english'))

		brown_ic = wordnet_ic.ic('ic-brown.dat')
		semcor_ic = wordnet_ic.ic('ic-semcor.dat')
		for e in self.inputEntities:
			# no blacklist, no 1-character entities, no only numbers, no entities that start with a number, no entities with more than 7 tokens
			if e in self.blacklist or len(e) <= 2 or e.isdigit() or e[0].isdigit() or len(nltk.word_tokenize(e)) >= 7:#
				self.invalidEntities.add(str(e))
				continue

			# no entities made only of stopwords and/or blacklist tokens (e.g. "a methodology")
			tokens = e.lower().split()  # Tokenize and convert to lowercase
			filtered_tokens = [t for t in tokens if t not in swords and t not in self.blacklist]
			if not filtered_tokens:  # Keep only if there is at least one valid token
				self.invalidEntities.add(str(e))
				continue

			if e in self.csoTopics:
				self.validEntities.add(e)
			elif e in self.magTopics:
				self.validEntities.add(e)
			elif e in self.wikidata_concepts.keys():
				self.validEntities.add(e)

			else:
				valid = True
				for synset in wn.synsets(e):
					ic_value = semcor_ic['n'][synset.offset()]
					if ic_value <= 4 and ic_value > 0:
						valid = False
						self.invalidEntities.add(str(e))
						break
				if valid:
					self.validEntities.add(e)

	def run(self):
		self.loadCSOTopics()
		self.loadBlacklist()
		self.loadMAGTopics()
		self.load_open_alex_concepts()
		self.validation()
		# Writing the valid entities to a CSV file
		with open(os.path.join(self.debug_output_dir,'valid_entities.csv'), mode="w", newline="") as file:
			writer = csv.writer(file)
			for item in self.validEntities:
				writer.writerow([item])
		with open(os.path.join(self.debug_output_dir,'invalid_entities.csv'), mode="w", newline="") as file:
			writer = csv.writer(file)
			for item in self.invalidEntities:
				writer.writerow([item])

	def get(self):
		#print(self.validEntities)
		return self.validEntities



if __name__ == '__main__':
	ev = EntitiesValidator(['computer science', 'danilo', 'svm', 'machine learning', 'fghjjjj', 'hello', 'method', 'methods', 'test', 'neural networks', 'a320', '320', '320a'], '../dataset/AI_whitelisted_parsed/')
	ev.run()
	#print(ev.get())











