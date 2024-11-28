def read_concerts(file_path):
    concerts = {}
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('#') and line.strip():
                genre, count = line.split(':')
                concerts[genre.strip()] = int(count.strip())
    concerts_list = [[item[0], item[1]] for item in concerts.items()]
    return concerts_list

def sort_concerts(concerts):
    sorted_concerts = sorted(concerts, key=lambda item: item[1], reverse=True)
    return sorted_concerts