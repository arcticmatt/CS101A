'''
Given directory of feature/label csvs, creates a training csv and 
test csv from randomly selected rows.
'''
import csv
import os
import random
import sys


TRAIN_PCT = 0.7


def is_train():
    return random.random() < TRAIN_PCT


def create_train_test(csv_directory, train_fname, test_fname):
    '''
    Takes in 
      - directory which holds csv files to "combine" into train/test sets
      - filename to write training set to
      - filename to write test set to 

    First, we do error checking. Then, for each csv file in the directory, we 
    iterate through all its rows. For each row, we flip a weighted coin 
    (is_train()), and depending on this result, we write the row to either
    the training set or the test set.
    '''

    if not os.path.exists(csv_directory):
        print 'csv_directory {} does not exist'.format(csv_directory)
        sys.exit()

    if not os.path.isdir(csv_directory):
        print 'csv_directory {} is not a valid directory'.format(csv_directory)
        sys.exit()

    if not os.listdir(csv_directory):
        print 'csv_directory {} is empty'.format(csv_directory)
        sys.exit()
    
    # Iterate through directory entries, and write rows to training/test csvs
    with open(train_fname, 'w') as train, open(test_fname, 'w') as test:
        train_writer = csv.writer(train, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        test_writer = csv.writer(test, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for filename in os.listdir(csv_directory):
            if filename.endswith('.csv'):
                # Open .csv file, iterate through every row. For each row, 
                # write it to train/test, depending on weighted coin flip
                with open(os.path.join(csv_directory, filename), 'r') as csv_read:
                    reader = csv.reader(csv_read, delimiter=',', quotechar='|')
                    for row in reader:
                        if is_train():
                            train_writer.writerow(row)
                        else:
                            test_writer.writerow(row)


if __name__ == '__main__':
    # Error checking
    if len(sys.argv) < 4:
        print 'usage: python create_train_test.py csv_directory train_fname test_fname'
        sys.exit()

    csv_directory = sys.argv[1]
    train_fname = sys.argv[2]
    test_fname = sys.argv[3]
    random.seed()

    create_train_test(csv_directory, train_fname, test_fname)
