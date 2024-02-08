import csv
import glob

# List of CSV files to be merged
csv_files = glob.glob('data/*.csv')

# Create the output file
with open('all_real_estates.csv', 'w', newline='') as f_out:
    writer = csv.writer(f_out)

    # Write the header of the first file to the output file
    with open(csv_files[0], 'r') as f_in:
        reader = csv.reader(f_in)
        headers = next(reader)
        writer.writerow(headers)

    # Write the contents of each file to the output file
    for csv_file in csv_files:
        with open(csv_file, 'r') as f_in:
            reader = csv.reader(f_in)
            # Skip the header of each file
            next(reader)
            for row in reader:
                writer.writerow(row)