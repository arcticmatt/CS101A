import boto3
import urllib

#before running this script, run aws configure with aws cli

# Let's use Amazon S3
s3 = boto3.resource('s3')

def upload(filepath, filename):
    data = open(filepath, 'rb')
    print s3.Bucket('kanyeblessed-s3').put_object(Key=filename, Body=data)

# Print out bucket names
def print_buckets():
    for bucket in s3.buckets.all():
        print(bucket.name)

def download_song(url, filepath):
    urllib.urlretrieve(url, filepath)

def send_to_s3(preview_url, save_to_filepath):
    download_song(preview_url, save_to_filepath)
    upload(save_to_filepath, save_to_filepath[save_to_filepath.rfind('/')+1:])

send_to_s3('http://d318706lgtcm8e.cloudfront.net/mp3-preview/f454c8224828e21fa146af84916fd22cb89cedc6', '/Users/kshitij/Documents/song2.mp3')
