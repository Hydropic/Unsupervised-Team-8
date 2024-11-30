import matplotlib.pyplot as plt

def visualize_concerts(concerts, per_time):
    """
    Input: sorted_concerts - a list of lists containing the genre and the number of concerts for each genre
           per_time - a string indicating the time period for which the number of concerts is calculated (e.g., 'year', 'two weeks')
    """
    # Extract genres and counts for plotting
    genres = [genre for genre, count in concerts]
    counts = [count for genre, count in concerts]

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