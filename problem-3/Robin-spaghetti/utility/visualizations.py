import matplotlib.pyplot as plt

def visualize_concerts(sorted_concerts, per_time):

    # Extract genres and counts for plotting
    genres = [genre for genre, count in sorted_concerts]
    counts = [count for genre, count in sorted_concerts]

    # Create the plot
    plt.figure(figsize=(12, 10))
    plt.barh(genres, counts, color='skyblue')
    plt.xlabel(f'Number of concerts per {per_time}')
    plt.ylabel('Genre')
    plt.title('Number of Concerts per Genre')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest values on top
    plt.yticks(fontsize=6)  # Adjust font size of genre names
    plt.tight_layout()  # Adjust layout to make room for labels
    plt.show()