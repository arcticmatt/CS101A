import requests
import csv
import base64
import collections
import time

CLIENT_ID = '4f7d3159a034481da00aa49e62f13d44'
CLIENT_SECRET = '4fccce79c4e643f2852db7ad12327b6a' # SHHHHH

YEARS = range(1960, 1970)
SONGS_PER_YEAR = 1000

SONG_FILENAME = 'song_data.csv'
START_OFFSET_FILENAME = 'end_offsets.csv'
END_OFFSET_FILENAME = 'end_offsets.csv'

start_offsets = {year: 0 for year in YEARS}
end_offsets = collections.OrderedDict()
LOAD_START_OFFSETS = False

access_token = None
access_token_expiry = None


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
        # base 64 encoded data
        self.binary_data = self.get_binary_song_data(preview_url)

    def get_binary_song_data(self, url):
        r = requests.get(url)
        # binary data
        song_data = r.content
        return base64.b64encode(song_data)


def create_song(song_req, album_obj):
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


def create_album(album_req):
    album_dict = album_req.json()
    try:
        album_date = album_dict['release_date']
        album_genres = album_dict['genres']
    except KeyError as e:
        print 'Error parsing album_req {}, error = {}'.format(album_req.text, e)
    return Album(album_date, album_genres)


def make_authorized_request(url, req_f):
    if time.time() > access_token_expiry:
        set_access_token()
    
    auth_field = 'Bearer ' + access_token

    return req_f(url, headers={'Authorization': auth_field})


def get_songs(year, starting_offset, num_songs=100):
    songs_for_year = []
    songs_proc = 0
    albums_proc = 0
    req_str = 'https://api.spotify.com/v1/search?q=year:{0}&type=album&offset={1}'.format(year, starting_offset)
    req = make_authorized_request(req_str, requests.get)

    while songs_proc < num_songs:
        print '{} albums processed, {} songs processed...'.format(albums_proc, songs_proc)
        # List of 20 albums. Each item is a dictionary containing some album
        # information (album_type, available_markets, external_urls, href, id,
        # images, name, type, uri). The item does not contain the full album
        # information (release date, genres, etc.). However, the href
        # can be used to fetch that.
        try:
            albums = req.json()['albums']['items']
        except KeyError as e:
            print 'Error parsing req {}, error = {}'.format(req.text, e)

        # For each album, iterate through songs until we come across a song
        # with a preview_url
        for album in albums:
            album_req = make_authorized_request(album['href'], requests.get)
            try:
                album_songs_req = make_authorized_request(album_req.json()['tracks']['href'],
                        requests.get)
            except KeyError as e:
                print 'Error parsing album_songs_req {}, error = {}'.format(album_req.text, e)
            songs = album_songs_req.json()['items']

            for song in songs:
                print 'Processing songs for album {}'.format(album_req.json()['name'].encode('utf-8'))
                if song['preview_url']:
                    # Get song information (so we can get song popularity).
                    # Note that a list of genres is given in the album_req.
                    album_obj = create_album(album_req)
                    song_req = make_authorized_request(song['href'], requests.get)
                    song_obj = create_song(song_req, album_obj)
                    songs_for_year.append(song_obj)
                    songs_proc += 1
                    if songs_proc == num_songs:
                        end_offsets[year] = albums_proc + req.json()['albums']['limit']
                        return songs_for_year
                    # print 'Preview url found for song {}'.format(song_obj.name)
                    break

        req = make_authorized_request(req.json()['albums']['next'], requests.get)
        albums_proc += req.json()['albums']['limit']
        print ''

    return songs_for_year


def load_start_offset_data(filename):
    offsets = collections.OrderedDict()
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['year', 'offset'])
        for row in reader:
            year = int(row['year'])
            offset = int(row['offset'])
            offsets[year] = offset
    return offsets


def write_end_offset_data(filename):
    with open(filename, 'w+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for year, end_offset in end_offsets.iteritems():
            writer.writerow([year, end_offset])


def write_song_data(filename, songs):
    '''
    SOLUTION 3: ENCODE LATE
    http://farmdev.com/talks/unicode/
    '''

    with open(filename, 'w+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for song in songs:
            writer.writerow([song.song_id, song.name.encode('utf-8'), song.year,
                             song.popularity, song.preview_url,
                             song.binary_data])


def set_access_token():
    global access_token 
    global access_token_expiry 

    auth_field = 'Basic ' + base64.b64encode(CLIENT_ID + ':' + CLIENT_SECRET)
    # Get access token using Spotify credentials
    access_token_req = requests.post("https://accounts.spotify.com/api/token", 
                                     data={'grant_type': 'client_credentials'},
                                     headers={'Authorization': auth_field})
    access_token = access_token_req.json()['access_token'] 
    print 'Using access token, valid for {} minutes'.format(
        access_token_req.json()['expires_in'] / 60.0)

    access_token_expiry = time.time() + 3600


if __name__ == '__main__':
    set_access_token()

    if LOAD_START_OFFSETS:
        start_offsets = load_start_offset_data(START_OFFSET_FILENAME) 

    # We'll use albums from the 60's to the 00's
    print 'Getting {} songs from {}'.format(SONGS_PER_YEAR * len(YEARS), YEARS)
    songs = []
    for year in YEARS:
        songs += get_songs(year=year, starting_offset=start_offsets[year], 
                           num_songs=SONGS_PER_YEAR)

    print 'Writing song data to CSV...'
    write_song_data(SONG_FILENAME, songs)

    print 'Writing end offset data to CSV...'
    write_end_offset_data(END_OFFSET_FILENAME)
