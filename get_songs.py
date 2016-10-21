import base64
import collections
import csv
import requests
import os
import sys
import time
from requests.exceptions import ConnectionError

SONG_FILENAME = 'song_data.csv'
OFFSET_FILENAME = 'end_offsets.csv'

MP3_SUBDIR = './mp3s/'
if not os.path.exists(MP3_SUBDIR):
    os.makedirs(MP3_SUBDIR)
CHUNK_SIZE = 1024 * 1024

def require(condition, error_msg):
    if not condition:
        print error_msg
        sys.exit(1)

# Used to store Spotify album information.
class Album:
    def __init__(self, album_date, album_genres):
        self.date = album_date
        self.genres = album_genres

# Used to store Spotify song information.
# Genres is currently not populated for a lot (basically any) albums. We
# won't actually use it for now.
class Song:
    def __init__(self, song_id, name, year, popularity, preview_url, genres):
        self.song_id = song_id
        self.name = name
        self.year = year
        self.popularity = popularity
        self.preview_url = preview_url
        self.genres = genres
        self.filename = MP3_SUBDIR + str(self.song_id) + '.mp3'

    def download_song(self):
        try:
            r = requests.get(self.preview_url)
            with open(self.filename, 'wb') as fd:
                for chunk in r.iter_content(CHUNK_SIZE):
                    fd.write(chunk)
            return True
        except ConnectionError as e:
            print 'Error downloading preview_url {}, error = {}'.format(self.preview_url, e)
            return False

    def persist(self, filename):
        print "Persisting song %s"%self.name
        with open(filename, 'a+') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            # Only save this song as a training point if we're able to download its preview
            if self.download_song():
                writer.writerow([self.song_id, ''.join(ch for ch in self.name.encode('utf-8') if ch.isalnum()), 
                                 self.year, self.popularity, self.preview_url,
                                 self.filename])

class AlbumOffsets:
    '''
    Class used to store/persist offsets of albums scraped for each year
    '''
    def __init__(self, filename):
        self.filename = filename
        self.offsets = self._load_offsets()

    def get(self, year):
        return self.offsets.get(year, 0)

    def _load_offsets(self):
        offsets = collections.OrderedDict()
        if not os.path.exists(self.filename):
            return {}

        with open(self.filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=['year', 'offset'])
            for row in reader:
                year = int(row['year'])
                offset = int(row['offset'])
                offsets[year] = offset
        return offsets

    def set(self, year, new_offset):
        self.offsets[year] = new_offset

    def write(self):
        with open(self.filename, 'w+') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for year, offset in self.offsets.iteritems():
                writer.writerow([year, offset])

    def update(self, year, new_offset):
        self.set(year, new_offset)
        self.write()

class SpotifyClient:
    '''
    Class for obtaining song data from the Spotify API
    '''
    def __init__(self, client_id, client_secret, offsets):
        self.client_id = client_id
        self.client_secret = client_secret
        self.set_access_token()
        self.offsets = offsets

    def _get_albums(self, year, offset):
        '''
        Returns 20 albums for the specified year, or an empty array if an error occurs.
        Starts searching at the specified offset
        '''
        req_str = 'https://api.spotify.com/v1/search?q=year:{0}&type=album&offset={1}'.format(year, offset)
        # List of 20 albums. Each item is a dictionary containing some album
        # information (album_type, available_markets, external_urls, href, id,
        # images, name, type, uri). The item does not contain the full album
        # information (release date, genres, etc.). However, the href
        # can be used to fetch that.
        req = self.make_authorized_request(req_str, requests.get)
        try:
            return req.json()['albums']['items']
        except KeyError as e:
            print 'Error parsing req {}, error = {}'.format(req.text, e)
            return []

    def _get_first_song_with_preview(self, album):
        album_req = self.make_authorized_request(album['href'], requests.get)
        album_songs_req = None
        try:
            print 'Processing songs for album {}'.format(album_req.json()['name'].encode('utf-8'))
            album_songs_req = self.make_authorized_request(album_req.json()['tracks']['href'],
                    requests.get)
            album_songs = album_songs_req.json()['items']
            for song in album_songs:
                if song['preview_url']:
                    # Get song information (so we can get song popularity).
                    # Note that a list of genres is given in the album_req.
                    album_obj = self.create_album(album_req)
                    song_req = self.make_authorized_request(song['href'], requests.get)
                    return self.create_song(song_req, album_obj)
        except KeyError as e:
            print 'Error parsing album_req {} or album_songs_req {}, error = {}'.format(
                album_req, album_songs_req, e)
        return None

    def get_song_batch(self, year):
        '''
        Get a batch of songs for the specified year. Specifically, makes a request
        for albums for the specified year starting from the corresponding offset.
        Returns a list consisting of one song (with a preview URL) for each such album
        (if an album does not contain a song with a preview URL, no song is included for
        that album).
        '''
        curr_offset = self.offsets.get(year)
        albums = self._get_albums(year, curr_offset)
        songs = []
        # For each album, iterate through songs until we come across a song
        # with a preview_url
        for album in albums:
            song_with_preview = self._get_first_song_with_preview(album)
            if song_with_preview is not None:
                songs.append(song_with_preview)

        self.offsets.update(year, curr_offset + len(albums))
        return songs


    def create_song(self, song_req, album_obj):
        song_dict = song_req.json()
        try:
            song_id = song_dict['id']
            song_name = song_dict['name']
            song_year = album_obj.date.split('-')[0]
            song_popularity = song_dict['popularity']
            song_preview_url = song_dict['preview_url']
        except KeyError as e:
            print 'Error parsing song_req {}, error = {}'.format(song_req.text, e)
        return Song(song_id, song_name, song_year, song_popularity, song_preview_url, album_obj.genres)

    def create_album(self, album_req):
        album_dict = album_req.json()
        try:
            album_date = album_dict['release_date']
            album_genres = album_dict['genres']
        except KeyError as e:
            print 'Error parsing album_req {}, error = {}'.format(album_req.text, e)
        return Album(album_date, album_genres)

    def set_access_token(self):
        auth_field = 'Basic ' + base64.b64encode(self.client_id + ':' + self.client_secret)
        # Get access token using Spotify credentials
        access_token_req = requests.post("https://accounts.spotify.com/api/token",
                                         data={'grant_type': 'client_credentials'},
                                         headers={'Authorization': auth_field})
        self.access_token = access_token_req.json()['access_token']
        print 'Using access token, valid for {} minutes'.format(
            access_token_req.json()['expires_in'] / 60.0)

        self.access_token_expiry = time.time() + 3600

    def make_authorized_request(self, url, req_f):
        if time.time() > self.access_token_expiry:
            self.set_access_token()
        # Repeatedly try to make the request, waiting between successive failed
        # requests.
        sleep_duration = 0
        while True:
            try:
                time.sleep(sleep_duration)
                auth_field = 'Bearer ' + self.access_token
                return req_f(url, headers={'Authorization': auth_field})
            except ConnectionError as e:
                sleep_duration = sleep_duration * 2 + 1
                print "HTTP request for %s failed with ConnectionError: %s"%(url, e)
                print "Retrying request after %s seconds"%(sleep_duration)


class Scraper:
    def __init__(self, client, song_data_csv_filename):
        self.client = client
        # Path to output file, where song data will be written
        self.song_data_csv_filename = song_data_csv_filename

    def scrape(self, start_year, end_year, songs_per_year):
        years = range(start_year, end_year)
        print 'Getting {} songs from {}'.format(songs_per_year * len(years), years)
        for year in years:
            print "Loading %s songs for year %s. %s loaded thus far."%(songs_per_year,
                year, self.client.offsets.get(year))
            self._get_songs_for_year(year=year, num_songs=songs_per_year)

    def _get_songs_for_year(self, year, num_songs):
        '''
        Get <num_songs> songs for the specified year
        '''
        songs = []
        songs_to_proc = num_songs - self.client.offsets.get(year)
        while songs_to_proc > 0:
            print '{} total songs processed for year {}...'.format(self.client.offsets.get(year), year)
            # Get a batch of songs for the current year
            song_batch = self.client.get_song_batch(year)
            batch_size = len(song_batch)
            # Persist each song in the current batch to our CSV
            for i in xrange(min(batch_size, songs_to_proc)):
                song_batch[i].persist(self.song_data_csv_filename)
            songs.extend(song_batch)
            songs_to_proc -= batch_size

def scrape(client, start_year, end_year, songs_per_year, song_data_csv_filename):
    '''
    Main scraping method.
    '''
    scraper = Scraper(client, song_data_csv_filename)
    scraper.scrape(start_year, end_year, songs_per_year)

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print "usage: python get_songs.py client_id client_secret start_year end_year songs_per_year"

    client_id = sys.argv[1]
    client_secret = sys.argv[2]
    start_year = int(sys.argv[3])
    end_year = int(sys.argv[4])
    songs_per_year = int(sys.argv[5])
    require(end_year > start_year, "End year must be greater than or equal to start year")

    album_offsets = AlbumOffsets(OFFSET_FILENAME)
    client = SpotifyClient(client_id, client_secret, album_offsets)
    scrape(client, start_year, end_year, songs_per_year, SONG_FILENAME)

