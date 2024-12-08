# Step 1: Open the input text file in read mode
with open('/Users/aymansoultana/unsupervised3/Unsupervised-Team-8/vaccinate_laplacian_eigen_map.txt', 'r') as infile:
    # Step 2: Read the content of the file
    content = infile.read()

# Step 3: Split the content by commas to get individual person IDs
person_ids = content.split(',')

# Step 4: Open a new output text file in write mode
with open('person_ids.txt', 'w') as outfile:
    # Step 5: Write each person ID to the new file, each on a new line
    for person_id in person_ids:
        outfile.write(person_id.strip() + '\n')