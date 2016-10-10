import boto3
import urllib
import requests
import csv
import base64

# Before Running this script
# pip install awscli
# aws configure
# enter credentials

# dependencies:
# pip install boto3

# Let's use Amazon S3
s3 = boto3.resource('s3')

def upload(filepath, filename):
    data = open(filepath, 'rb')
    print s3.Bucket('kanyeblessed-s3').put_object(Key=filename, Body=data)

# Print out bucket names
def print_buckets():
    for bucket in s3.buckets.all():
        print(bucket.name)

def download_to_filepath(url, filepath):
    urllib.urlretrieve(url, filepath)

def get_binary_song_data(url):
    r = requests.get(url)
    # binary data
    song_data = r.content
    return song_data

def send_to_s3(preview_url, save_to_filepath):
    download_to_filepath(preview_url, save_to_filepath)
    upload(save_to_filepath, save_to_filepath[save_to_filepath.rfind('/')+1:])

# CSV Utils
def open_csv_writer(filepath):
    f = open(filepath, 'wb')
    writer = csv.writer(f)
    return writer

def append_song_to_csv(csv_writer, song_id, song_data):
    csv_writer.writerow((song_id, base64.encodestring(song_data)))

def mainfunc():
    writer = open_csv_writer('test.csv')
    data = get_binary_song_data('http://d318706lgtcm8e.cloudfront.net/mp3-preview/f454c8224828e21fa146af84916fd22cb89cedc6')
    append_song_to_csv(writer, 1, data)

mainfunc()
