import matplotlib.pyplot as plt
from utility import visualizations, graph
import json
import csv
import numpy as np
from alive_progress import alive_bar

file_path = 'problem-3\\Data\\grupee_data\\n_concerts.txt'
concerts = graph.read_concerts(file_path)
# Perform the division and multiplication while maintaining the tuple structure
concert_per_two_weeks_scaler = [((vc[1] / 52.1429) * 2.0) for vc in concerts]
# visualizations.visualize_concerts(concert_per_two_weeks, "two weeks")
# print("visit per two weeks", concert_per_two_weeks)

# Extract the preferences of the grupees
with open('problem-3\Data\grupee_data\preferences.json', 'r') as file:
    preferences = json.load(file)

# Read the connections between grupees
friend_pairs = []
with open('problem-3\\Data\\grupee_data\\friends.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        friend_pairs.append(row)
    friend_pairs.pop(0) # first line contains an unnecessary comment

# probability to infect a friend depending on preferences
both_like = 393/1000
one_like = 18/1000
neither_like = 2/1000
genre_count = len(preferences[friend_pairs[0][0]]) 
risk_per_id = [[0] * genre_count for _ in range(len(preferences.keys()))]
with alive_bar(len(friend_pairs), title='Processing Friend IDs') as bar:
    for f_id in friend_pairs: # f_id is a list of two friend ids
        risk_per_id_row = [0] * genre_count
        pref_1 = preferences[f_id[0]]
        pref_2 = preferences[f_id[1]]
        for i, (p1, p2) in enumerate(zip(pref_1, pref_2)): # iterate over the prefereed genres of the two friends
            if p1 == '1' and p2 == '1':
                risk_per_id_row[i] += both_like
            elif p1 == '1' or p2 == '1':
                risk_per_id_row[i] += one_like
            else:
                risk_per_id_row[i] += neither_like
        # Element wise addition of the risk per genre
        risk_per_id[int(f_id[0])] = [x + y for x, y in zip(risk_per_id[int(f_id[0])], risk_per_id_row)]
        risk_per_id[int(f_id[1])] = [x + y for x, y in zip(risk_per_id[int(f_id[1])], risk_per_id_row)]
        bar()
# scale the risk per genre of each grupee by the number of concerts per two weeks
scaled_risk_per_id = [[x * y for x, y in zip(row, concert_per_two_weeks_scaler)] for row in risk_per_id] # theoretically could scale by concerts instead
# Sum the elements of each row in scaled_risk_per_ids_concert
sum_scaled_risk_per_id = [sum(row) for row in scaled_risk_per_id]
# Calculate the number of top elements to select (12% of the total length)
top_percentage = 0.12
num_top_elements = int(len(sum_scaled_risk_per_id) * top_percentage)

# Get the indices of the top elements
top_indices = np.argsort(sum_scaled_risk_per_id)[-num_top_elements:][::-1]

# Get the values of the top elements
top_values = [sum_scaled_risk_per_id[i] for i in top_indices]

print("Top 12% indices:", top_indices)
print("Top 12% values:", top_values)
# top 12% of the riskiest people to be affected by the virus (due to visitation of concerts with friends)
# FROM HERE ON: 
# top_indices: list of ids of the top 12% of the riskiest people to be affected by the virus (sorted by risk)
# top_values: list of the corresponding risks of the top 12% of the riskiest people to be affected by the virus