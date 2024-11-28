import matplotlib.pyplot as plt
from utility import visualizations, graph
import json
import csv
import numpy as np
from alive_progress import alive_bar

file_path = 'problem-3\\Data\\grupee_data\\n_concerts.txt'
concerts = graph.read_concerts(file_path)
# Perform the division and multiplication while maintaining the tuple structure
concert_per_two_weeks = [(vc[0], (vc[1] / 52.1429) * 2.0) for vc in concerts]
# visualizations.visualize_concerts(concert_per_two_weeks, "two weeks")
# print("visit per two weeks", concert_per_two_weeks)

# Extract the preferences of the grupees
with open('problem-3\Data\grupee_data\preferences.json', 'r') as file:
    preferences = json.load(file)

risk_per_ids = [0] * len(preferences.keys()) # Initialize the risk per id list with zeros (index corresponds to the id, value to the risk)

# Read the connections between grupees
friend_ids = []
with open('problem-3\\Data\\grupee_data\\friends.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        friend_ids.append(row)
    friend_ids.pop(0) # first line contains an unnecessary comment

both_like = 393/1000
one_like = 18/1000
neither_like = 2/1000
with alive_bar(len(friend_ids), title='Processing Friend IDs') as bar:
    for f_id in friend_ids: # TODO: consider concerts per year for each friend
        pref_1 = preferences[f_id[0]]
        pref_2 = preferences[f_id[1]]
        for p1, p2 in zip(pref_1, pref_2):
            if p1 == 1 and p2 == 1:
                risk_per_ids[int(f_id[0])] += both_like
                risk_per_ids[int(f_id[1])] += both_like
            elif p1 == 1 or p2 == 1:
                risk_per_ids[int(f_id[0])] += one_like
                risk_per_ids[int(f_id[1])] += one_like
            else:
                risk_per_ids[int(f_id[0])] += neither_like
                risk_per_ids[int(f_id[1])] += neither_like
        bar()
# Calculate the number of top elements to select (12% of the total length)
top_percentage = 0.12
num_top_elements = int(len(risk_per_ids) * top_percentage)

# Get the indices of the top elements
top_indices = np.argsort(risk_per_ids)[-num_top_elements:][::-1]

# Get the values of the top elements
top_values = [risk_per_ids[i] for i in top_indices]

print("Top 12% indices:", top_indices)
print("Top 12% values:", top_values)
# top 12% of the riskiest people to be affected by the virus (due to visitation of concerts with friends)
# NOTE: currently this does not take the amount of different concerts per year into consideration
# FROM HERE ON: 
# top_indices: list of ids of the top 12% of the riskiest people to be affected by the virus (sorted by risk)
# top_values: list of the corresponding risks of the top 12% of the riskiest people to be affected by the virus