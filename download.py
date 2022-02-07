# Dependencies
import urllib.request
import urllib.parse
import argparse
import tqdm
import time
import json
import os

#  Define main
if __name__ == '__main__':

    # Initialize argument parser
    parser = argparse.ArgumentParser(description='download images from JSON-list file')
    # Define arguments
    parser.add_argument('-i', '--in_path', required=True, help='Define path to input file', type=str)
    parser.add_argument('-o', '--out_dir', required=True, help='Define path to output directory', type=str)
    parser.add_argument('-t', '--delay', default=0.1, help='Define time interval between images download', type=float)
    # Parse arguments
    args = parser.parse_args()

    # Define path to output directory
    out_path = args.out_dir
    # Create output directory, if any
    os.makedirs(out_path, exist_ok=True)
    # Open input file
    with open(args.in_path, 'r') as file:
        # Initialize counter
        counter = 0
        # Loop through each line in file
        for line in tqdm.tqdm(file, desc='Downloading images'):
            # Try executing
            try:
                # Parse line as JSON object
                line = json.loads(line.strip())
                # Retrieve image ID and URL from JSON object
                identifier, url = line.get('id'), line.get('url')
                # Clean url of query strung
                cleared = urllib.parse.urljoin(url, urllib.parse.urlparse(url).path)
                # Retrieve extension out of image path
                _, extension = os.path.splitext(cleared)
                # Download image from url
                urllib.request.urlretrieve(url, os.path.join(out_path, identifier) + extension)
                # Update counter
                counter = counter + 1
            # Do nothing
            except json.decoder.JSONDecodeError as e:
                continue
            # Wait for given time
            time.sleep(args.delay)

    # Signal termination
    print('Downloaded %d images' % counter)
