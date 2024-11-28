from utility import visualizations, graph

file_path = 'problem-3\\Data\\grupee_data\\n_concerts.txt'
concerts = graph.read_concerts(file_path)
visitors_concert = graph.sort_concerts(concerts)
print(visitors_concert)
visualizations.visualize_concerts(visitors_concert, "year")
# OUTPUT: visualize most visited concerts

visitors_concert_per_two_weeks = [(vc[0], (vc[1] / 52.1429) * 2.0) for vc in visitors_concert]
visualizations.visualize_concerts(visitors_concert_per_two_weeks, "two weeks")
#OUTPUT: visualize most visited concerts per two weeks