import pandas as pd
import numpy as np

import codecs
import json
import sys
import re

import plotly
plotly.tools.set_credentials_file(username='SKsaqlain',api_key='4aQeHDqR85H7d8POxmmR')
# from plotly.offline import init_notebook_mode, iplot
# init_notebook_mode(connected=True)
import plotly.plotly as  py
import plotly.figure_factory as ff
import plotly.graph_objs as go

class ExampleEvaluator:
	def __init__(self, answer_file_path, round=1):
		"""
		`round` : Holds the round for which the evaluation is being done. 
		can be 1, 2...upto the number of rounds the challenge has.
		Different rounds will mostly have different ground truth files.
		"""
		self.answer_file_path = answer_file_path
		self.round = round
		#Identifier to store protected variables.
		self.variables=dict()
	def f1(self,p, r):
		if r == 0.:
			return 0.
		return 2 * p * r / float(p + r)


	def strict(self,true_and_prediction):
	    num_entities = len(true_and_prediction)
	    correct_num = 0.
	    for true_labels, predicted_labels in true_and_prediction:
	        correct_num += set(true_labels) == set(predicted_labels)
	    precision = recall = correct_num / num_entities
	    return {'p': precision,
	            'r': recall,
	            'f1': self.f1(precision, recall)}


	def loose_macro(self,true_and_prediction):
	    num_entities = len(true_and_prediction)
	    p = 0.
	    r = 0.
	    for true_labels, predicted_labels in true_and_prediction:
	        if len(predicted_labels) > 0:
	            p += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
	        if len(true_labels):
	            r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
	    precision = p / num_entities
	    recall = r / num_entities
	    return {'p': precision,
	            'r': recall,
	            'f1': self.f1(precision, recall)}


	def loose_micro(self,true_and_prediction):
	    num_predicted_labels = 0.
	    num_true_labels = 0.
	    num_correct_labels = 0.
	    for true_labels, predicted_labels in true_and_prediction:
	        num_predicted_labels += len(predicted_labels)
	        num_true_labels += len(true_labels)
	        num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
	    precision = num_correct_labels / num_predicted_labels
	    recall = num_correct_labels / num_true_labels
	    return {'p': precision,
	            'r': recall,
	            'f1': self.f1(precision, recall)}


	def get_annotations(self,file_name):
	    annotations_map = {}

	    # read the file, check for correct format
	    with codecs.open(file_name, 'r', 'utf-8') as f:
	        json_data = None
	        try:
	            json_data = json.loads(f.read())
	        except Exception as e:
	            print("Exception - " + str(e))
	        for k, v in json_data.items():
	            results = v
	        if (len(results) < 1):
	            print("Invalid submission file.")
	            sys.exit() 
	             
	        # now we'll read each json element
	        for result in results:
	            try:
	                if ('text' not in result or 'entity' not in result or 'types' not in result):
	                    print("Invalid result json element.")
	                    print(result)
	                    continue
	            except Exception as e:
	                print("Exception - " + str(e))
	            

	    		
	            anonymized_text = self.get_anonymized_text(result['text'], result['entity'])
	            annotations_map[anonymized_text] = (result['entity'], result['types'])

	        return annotations_map
	      
	def get_anonymized_text(self,text, entity=None):
	    if entity is None or len(entity) < 1:
	        return text
	    patterns = [r'\b%s\b' % re.escape(entity), r'\bXXXX XXXX XXXX XXXX\b', r'\bXXXX XXXX XXXX\b', r'\bXXXX XXXX\b', r'\bXXXX\b', r'\bXX/XX/XXXX\b', r'\bXX/XX/\b']
	    regex_patterns = '|'.join(patterns)
	    try:
	        mentions = re.findall(regex_patterns, text)
	        if (len(mentions) == 0):
	            return text
	        elif (len(mentions) != 1):
	            print("Error! Only one entity mention per sentence should be present.")
	            print(text)
	            print(mentions)
	            return text
	        mention = mentions[0]
	        entity = 'XXXX XXXX'
	        text = text.replace(mention, entity)
	    except:
	        return text

	    return text

	def _evaluate(self, client_payload, _context={}):
		"""
		`client_payload` will be a dict with (atleast) the following keys :
		- submission_file_path : local file path of the submitted file
		- aicrowd_submission_id : A unique id representing the submission
		- aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
		"""
		submission_file_path = client_payload["submission_file_path"]
		aicrowd_submission_id = client_payload["aicrowd_submission_id"]
		aicrowd_participant_uid = client_payload["aicrowd_participant_id"]

		ground_truth_file = self.answer_file_path
		input_file = submission_file_path

		# we'll load the input files to maps
		# key is anonymized, and hence normalized text
		# values are tuples containing (entity, labels_str)
		# labels_str contain types separated by space
		ground_truths_map = self.get_annotations(ground_truth_file)
		submissions_map = self.get_annotations(input_file)

		true_and_prediction = []
		for text, gt_tuple in ground_truths_map.items():
			if text in submissions_map.keys():
				#print(text)
				# each tuple has the entity and its type
				true_labels_str = gt_tuple[1]
				true_labels = true_labels_str.split()
				#print(true_labels)
				annotation_tuple = submissions_map[text]
				predicted_labels_str = annotation_tuple[1]
				predicted_labels = predicted_labels_str.split()

				true_and_prediction.append((true_labels, predicted_labels))

		if len(true_and_prediction) < 1:
			print("Error: invalid prediction output")
			sys.exit()
		#preparing data for plotting
		for text in submissions_map.keys():
			#each tuple has the entity and its type
			annotation_tuple = submissions_map[text]
			type_name=annotation_tuple[1].split("/")[-1]
			
			entity_name=annotation_tuple[0]
			
			'''
			Each key in the "variables" identifier is the  protected variables such as 'location','profession',...etc
			The values associated to each key is another dict which will store all the items and the frequency respectively
			such as in the  case  of 'location' it might be 'india:20','usa:10,...etc.

			'''
			
			if(type_name not in self.variables.keys()):
				self.variables[type_name]=dict()
				self.variables[type_name][entity_name]=1

			else:
				if(entity_name not in self.variables[type_name].keys()):
					self.variables[type_name][entity_name]=1
				else:
					self.variables[type_name][entity_name]+=1
		metrics = {
		'strict': self.strict(true_and_prediction),
		'macro': self.loose_macro(true_and_prediction),
		'micro': self.loose_micro(true_and_prediction),
		}

		geometricMean = (metrics['strict']['f1'] * metrics['macro']['f1'] * metrics['micro']['f1']) ** (1 / 3)
		hist_data=[]
		group_labels=[]

		#pre-processing 'variables' for plots.
		data_trace=[]
		for ele in self.variables.keys():
			freq=[]
			entity_name_list=[]
			group_labels.append(ele)
			for x in self.variables[ele].keys():
				entity_name_list.append(x)
				freq.append(self.variables[ele][x])
			hist_data.append(freq)
			data_trace.append(go.Bar(x=entity_name_list,y=freq,name=ele))
		
		# for i in range(len(group_labels)):
		# 	print(group_labels[i],hist_data[i])
		# 	break
		#appending a bias factor to overcome singularity problem
		for data in hist_data:
			data.append(max(data)-0.01)
		fig=ff.create_distplot(hist_data,group_labels)
		
		_result_object={"score":{"metric":metrics,"geometricMean":geometricMean},"graphs":{"hist":fig,"bar":data_trace}}
		return _result_object

if __name__ == "__main__":
    # Lets assume the the ground_truth is a json file
    # and is present at data/ground_truth.json
    # and a sample submission is present at data/sample_submission.json
    # answer_file_path = "data/ground_truth.json"
    # _client_payload = {}
    # _client_payload["submission_file_path"] = "data/sample_submission.json"

    answer_file_path = "data/sample_ground_truth.json"
    _client_payload = {}
    _client_payload["submission_file_path"] = "data/sample_submission_file1.json"
    _client_payload["aicrowd_submission_id"] = 1123
    _client_payload["aicrowd_participant_id"] = 1234
    
    # Instaiate a dummy context
    _context = {}
    # Instantiate an evaluator
    aicrowd_evaluator = ExampleEvaluator(answer_file_path)
    # Evaluate
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print(result['score'])
    py.iplot(result['graphs']['hist'],filename='histogram')
    py.iplot(result['graphs']['bar'],filename='bar chart')
