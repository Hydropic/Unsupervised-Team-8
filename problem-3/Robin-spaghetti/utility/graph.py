def read_concerts(file_path):
    """
    Input: file_path - the path to the file containing the concerts data
    Output: concerts_list - a list of lists containing the genre and the number of concerts for each genre 
    """
    concerts = {}
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('#') and line.strip():
                genre, count = line.split(':')
                concerts[genre.strip()] = int(count.strip())
    concerts_list = [[item[0], item[1]] for item in concerts.items()]
    return concerts_list

def sort_concerts(concerts):
    """
    Input: concerts - a list of lists containing the genre and the number of concerts for each genre
    Output: sorted_concerts - a list of lists containing the genre and the number of concerts for each genre, sorted by the number of concerts in descending order
    """
    sorted_concerts = sorted(concerts, key=lambda item: item[1], reverse=True)
    return sorted_concerts