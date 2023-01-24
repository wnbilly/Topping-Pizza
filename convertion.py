import re
import json


regex = r"Total Epoch (\d*)\n\nPerformance per label : \[\[(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*\]\]\nMean of performance : (\d*\.\d*)\n\s*precision\s*recall\s*f1-score\s*support\n\n\s*([a-z]*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*\d*\n\s*([a-z]*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*\d*\n\s*([a-z]*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*\d*\n\s*([a-z]*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*\d*\n\s*([a-z]*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*\d*\n\s*([a-z]*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*\d*\n\s*([a-z]*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*\d*\n\s*([a-z]*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*\d*\n\s*([a-z]*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*\d*\n\s*([a-z]*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*\d*\n\n.+\n.+\n.+\n.+"

nb_label=10

regex = re.compile(regex)


dico = {}

with open('training_performance_evolution.txt', 'r') as f:

	for text in regex.findall(f.read()):
		idx = text[0]
		
		print(f"Epoch {idx}")
		
		dico[idx] = {}
		dico[idx]["mean of performance"] = float(text[1 + nb_label])
		
		dico[idx]["performance"] = {}
		dico[idx]["precision"] = {}
		dico[idx]["recall"] = {}
		dico[idx]["f1-score"] = {}
		
		for i in range(nb_label):
			
			
			label_name = text[1 + nb_label + 1 + i*4]
			
			dico[idx]["performance"][label_name] = float(text[1 + i])
			dico[idx]["precision"][label_name] = float(text[1 + nb_label + 2 + i*4])
			dico[idx]["recall"][label_name] = float(text[1 + nb_label + 3 + i*4])
			dico[idx]["f1-score"][label_name] = float(text[1 + nb_label + 4 + i*4])
				

with open('training_performance_evolution.json', 'w') as f:
	f.write(json.dumps(dico, indent=4))

