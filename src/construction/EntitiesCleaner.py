import json

from nltk.corpus import stopwords
from urllib.parse import unquote
import nltk
import csv
import re
import os

class EntitiesCleaner:

	def __init__(self, entities):
		
		self.entities = entities
		self.entity2cleaned_entity = {}
		self.csoResourcePath = '../../resources/CSO.3.1.csv'
		self.debug_output_dir = '../../outputs/extracted_triples/debug/'


	def cleanPunctuactonStopwords(self):
		swords = set(stopwords.words('english'))
		#regex pattern for chinese/japanese/Korean characters
		cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\u31f0-\u31ff\u3000-\u303f]')
		regex_puntuaction_ok = re.compile('[%s]' % re.escape("\"_`.'")) # possible characters
		regex_acronym = re.compile("\(.*") # remove acronyms e.g., machine learning (ml) -> machine learning
		puntuaction_reject = list("!#$%*+,/:;<=>?@=[]^{|}~/{}") + ['\\']
		for e in self.entities:
			if e.lower() not in swords:
				valid_puntuaction = True
				for c in e:
					if c in puntuaction_reject or cjk_pattern.search(c):
						valid_puntuaction = False
						#print('discard entity with invalid punctuation: ' + str(e))
						break

				if valid_puntuaction:
					e_fixed = e.replace('`', '').replace('\'s', '').replace('’s', '').replace('’', '').replace('\'', '').replace('(', '').replace(')', '').replace('.', '').replace('“', '').replace('”','').replace(' – ','-').replace(' - ','-').strip()
					e_fixed = regex_acronym.sub('', e_fixed).strip()
					e_fixed = regex_puntuaction_ok.sub('_', e_fixed)
					e_fixed = re.sub(r'\s+', ' ', e_fixed)
					e_fixed = e_fixed.lower()
					
					self.entity2cleaned_entity[e] = e_fixed


	def lemmatize(self):
		wnl = nltk.stem.WordNetLemmatizer()
		for e_original in list(self.entity2cleaned_entity.keys()):
			e_cleaned = self.entity2cleaned_entity[e_original]
			e_cleaned_tokens = nltk.word_tokenize(e_cleaned)
			if len(e_cleaned_tokens) >= 1:
				e_lemmatized = ' '.join(e_cleaned_tokens[:-1] + [wnl.lemmatize(e_cleaned_tokens[-1].strip(), 'n')])
				self.entity2cleaned_entity[e_original] = e_lemmatized

		#print('self.entity2cleaned_entity')
		#print(self.entity2cleaned_entity)


	def toPreferredString(self):
		csoTopics2preferredLabel = {}		
		with open(self.csoResourcePath, 'r', encoding='utf-8') as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			for row in csv_reader:
				r = unquote(row[1])

				if r == '<http://cso.kmi.open.ac.uk/schema/cso#preferentialEquivalent>':	
					t1 = unquote(row[0]).replace('<https://', '')[:-1]
					t2 = unquote(row[2]).replace('<https://', '')[:-1]
					if t1.startswith('cso.kmi.open.ac.uk/topics/') and t2.startswith('cso.kmi.open.ac.uk/topics/'):
						t1 = t1.split('/')[-1]
						t1 = t1.lower().replace('_',' ')
						t2 = t2.split('/')[-1]
						t2 = t2.lower().replace('_',' ')
						csoTopics2preferredLabel[t1] = t2
						
			for e_original in self.entity2cleaned_entity:
				e_cleaned = self.entity2cleaned_entity[e_original]
				if e_cleaned in csoTopics2preferredLabel: 
					self.entity2cleaned_entity[e_original] = csoTopics2preferredLabel[e_cleaned]

	def run(self):
		#print input entities before cleaning for debugging:
		with open(os.path.join(self.debug_output_dir, 'inputEntitiesFromExtraction.csv'), mode="w", newline="") as file:
			writer = csv.writer(file)
			for item in self.entities:
				writer.writerow([item])

		self.cleanPunctuactonStopwords()
		self.lemmatize()
		with open(os.path.join(self.debug_output_dir, 'entity2cleaned_entity_after_lemmatization_1.json'), 'w', encoding="utf-8") as fw1:
			json.dump(self.entity2cleaned_entity, fw1, indent=4, ensure_ascii=False)
		self.toPreferredString()
		self.lemmatize()
		with open(os.path.join(self.debug_output_dir, '/entity2cleaned_entity_after_lemmatization_2.json'), 'w', encoding="utf-8") as fw2:
			json.dump(self.entity2cleaned_entity, fw2, indent=4, ensure_ascii=False)

	def get(self):
		return self.entity2cleaned_entity

if __name__ == '__main__':
	ec = EntitiesCleaner(['artificial neural network', 'artificial neural networks', 'back propagation neural networks', 'back-propagation neural network', 'back-propagation neural networks', 'neural network', 'neural network model', '`` Machine Learning (ML)', 'dynamically fused graph network ( dfgn )', 'programming languages', 'python', 'sparql\'s queries', 'it', 'IT', 'rough set', 'programming languages', 'non-rigid registration', 'computer science ontology ( CSO', 'neural networks'])
	ec.run()
	#print(ec.get())
		





