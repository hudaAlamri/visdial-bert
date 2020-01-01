# parse charades datasets to get the actions set

import csv

data_all = {}
input_file = csv.DictReader(open("../data/Charades/Charades_v1_train.csv"))
actions_list = {}

for row in input_file:
	id = row['id']
	data_all[id] = row
	action_list = [v.split(' ')[0] for v in row['actions'].split(';')]
	actions_list[id] = action_list


with open("../data/Charades/Charades_v1_classes.txt") as f:
    actions_names = dict(x.rstrip().split(None,1) for x in f)


# given a video_id, retrieve the set of actions and their temporal locations:
vid_id = '9GS13'
l_names = [actions_names[each] for each in actions_list[vid_id]]
for each in l_names:
	print(each)


