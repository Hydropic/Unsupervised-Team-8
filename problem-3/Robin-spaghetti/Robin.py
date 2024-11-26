def read_and_sort_concerts(file_path):
    concerts = {}
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('#') and line.strip():
                genre, count = line.split(':')
                concerts[genre.strip()] = int(count.strip())
    
    sorted_concerts = sorted(concerts.items(), key=lambda item: item[1], reverse=True)
    return sorted_concerts

file_path = 'problem-3\\Data\\grupee_data\\n_concerts.txt'
sorted_concerts = read_and_sort_concerts(file_path)
for genre, count in sorted_concerts:
    print(f"{genre}: {count}")

import matplotlib.pyplot as plt

def read_and_sort_concerts(file_path):
    concerts = {}
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('#') and line.strip():
                genre, count = line.split(':')
                concerts[genre.strip()] = int(count.strip())
    
    sorted_concerts = sorted(concerts.items(), key=lambda item: item[1], reverse=True)
    return sorted_concerts

file_path = 'problem-3\\Data\\grupee_data\\n_concerts.txt'
sorted_concerts = read_and_sort_concerts(file_path)

# Extract genres and counts for plotting
genres = [genre for genre, count in sorted_concerts]
counts = [count for genre, count in sorted_concerts]

# Create the plot
plt.figure(figsize=(12, 10))
plt.barh(genres, counts, color='skyblue')
plt.xlabel('Number of Visitors')
plt.ylabel('Concerts')
plt.title('Number of Visitors per Concert Genre')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest values on top
plt.yticks(fontsize=6)  # Adjust font size of genre names
plt.tight_layout()  # Adjust layout to make room for labels
plt.show()

# OUTPUT: Plot of visitors per concert genre that might be interesting for the presentation