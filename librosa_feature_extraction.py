from __future__ import print_function
import librosa
import sys
import numpy as np
import glob
import os
import csv


MFCC_DATA_FILENAME = 'mfcc_data.csv'
NUM_COEFFS = 100
NUM_FRAMES = 1325
HOP_LENGTH = 500

# You should run this script from where the python code is located,
# i.e. the root folder of the project.

def write_header():
    fieldnames = ['label']
    fieldnames = fieldnames + range(NUM_FRAMES * NUM_COEFFS) + ['song_id']
    with open(MFCC_DATA_FILENAME, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def get_label(song_year):
    return int(song_year[2])

def write_mfcc_data_for_folder(read_file, folder_name):
    cwd = os.getcwd()
    wrong_len_count = 0
    with open(read_file, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                song_id = row[0]
                song_year = row[2]
                song_label = get_label(song_year)
                audio_path = cwd + '/' + folder_name + '/' + song_id + '.mp3'
                # load in the audio path
                y, sr = librosa.load(audio_path)

                # For a standard 30s preview, this should give us a NUM_COEFFS x NUM_FRAMES array
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=NUM_COEFFS, hop_length=HOP_LENGTH)
                mfcc_all = mfcc.flatten()
                mfcc_str = ','.join(map(str, mfcc_all))

                if len(mfcc_all) != NUM_COEFFS * NUM_FRAMES:
                    print ('wrong length for song path: ' + audio_path)
                    wrong_len_count += 1
                else:
                    with open(MFCC_DATA_FILENAME, 'a+') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',')
                        writer.writerow([song_label] + list(mfcc_all) + [song_id])

    print (str(wrong_len_count) + " songs had the wrong length")


print ("== Should be run from project root directory == ")

write_header_flag = raw_input('Write header to mfcc_data.csv (y/n)? ')
if write_header_flag == 'y':
    write_header()
    print ("== Header written == ")

folder_name = raw_input('folder name to pull mp3s from w/o slashes? (eg mp3s): ')
read_file = raw_input('file to read song data from with extension? (eg song_data.csv): ')
write_mfcc_data_for_folder(read_file, folder_name)
