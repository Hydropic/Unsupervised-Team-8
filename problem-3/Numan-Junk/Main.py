import json
import numpy as np
import csv
from alive_progress import alive_bar
# Give weight to Preferences (based on the concert count) -> then based on the preferences with weights, evaluate some value for each friedship-> find top %12 with highest frienship numbers 

######################################### Import Data #########################################
__path__ = 'Unsupervised-Team-8/problem-3/Data/grupee_data/'
n_concerts = {}
with open(__path__+'n_concerts.txt') as f:
    for line in f:
        if ':' in line:
            genre, count = line.strip().split(':', 1)
            n_concerts[genre] = int(count)
file_path = __path__+'preferences.json'
with open(file_path, 'r') as file:
    data = json.load(file)
# print(data)
# print(n_concerts)
friend_pairs = []
with open('Unsupervised-Team-8/problem-3/Data/grupee_data/friends.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        friend_pairs.append(row)
    friend_pairs.pop(0) # first line contains an unnecessary co

################################################################################################
def apply_weights(data):
    if data == 1:
        return 393/1000
    elif data == 0.5:
        return 18/1000
    elif data == 0:
        return 2/1000
# ADD MORE CARE INTO CONCERT COUNT - HOW MANY PEOPLE WHO ARE NOT FRIENDS TEND TO VISIT THE SAME CONCERT (Increase the weights for those genres which people like more in general)
def friendship_score(user1, user2):
    temp_friendship = {}
    binary1 = data[user1]
    binary2 = data[user2]
    for index, (b1, b2) in enumerate(zip(binary1, binary2)):
        average = (int(b1) + int(b2)) / 2
        weigted_rates = apply_weights(average)
        temp_friendship[index] = weigted_rates  # Use the index as the key to preserve order
    
    return temp_friendship
    
def weighted_friendship_vector(user1, user2):
    temp_friendship = friendship_score(user1, user2)
    vector_amplitude = 0
    weighted_vector = np.array([temp_friendship[index] * n_concerts[list(n_concerts.keys())[index]] for index in temp_friendship])
    vector_amplitude = np.sqrt(np.sum(np.square(weighted_vector)))
    return weighted_vector, vector_amplitude

id1 = '83'
id2 = '8310'
risk_per_id = np.zeros(len(data.keys()))
with alive_bar(len(friend_pairs), title='Processing Friend IDs') as bar:
    for f_id in friend_pairs: # f_id is a list of two friend ids
        risk_per_id_row = [0] * len(n_concerts.keys())

        weighted_vector, vector_amplitude = weighted_friendship_vector(f_id[0], f_id[1])
        # Element wise addition of the risk per genre
        risk_per_id[int(f_id[0])] = vector_amplitude
        risk_per_id[int(f_id[1])] = vector_amplitude

        bar()   
        top_percentage = 0.12
num_top_elements = int(len(risk_per_id) * top_percentage)

# Get the indices of the top elements
top_indices = (np.argsort(risk_per_id))[-num_top_elements:][::-1]

# Get the values of the top elements
top_values = [risk_per_id[i] for i in top_indices]

print("Top 12% indices:", top_indices)
print(("Top 12% values:", top_values))

