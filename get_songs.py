import requests
import csv
import base64

starting_offset = 0


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
    song_id = song_dict['id']
    song_name = song_dict['name']
    song_year = album_obj.date.split('-')[0]
    song_popularity = song_dict['popularity']
    song_preview_url = song_dict['preview_url']
    return Song(song_id, song_name, song_year, song_popularity, song_preview_url, album_obj.genres)


def create_album(album_req):
    album_dict = album_req.json()
    album_date = album_dict['release_date']
    album_genres = album_dict['genres']
    return Album(album_date, album_genres)


def get_songs(year, starting_offset, num_songs=100):
    songs_for_year = []
    songs_proc = 0
    albums_proc = 0
    req_str = 'https://api.spotify.com/v1/search?q=year:{0}&type=album&offset={1}'.format(year, starting_offset)
    req = requests.get(req_str)

    while songs_proc < num_songs:
        print '{} albums processed, {} songs processed...'.format(albums_proc, songs_proc)
        # List of 20 albums. Each item is a dictionary containing some album
        # information (album_type, available_markets, external_urls, href, id,
        # images, name, type, uri). The item does not contain the full album
        # information (release date, genres, etc.). However, the href
        # can be used to fetch that.
        albums = req.json()['albums']['items']

        # For each album, iterate through songs until we come across a song
        # with a preview_url
        for album in albums:
            album_req = requests.get(album['href'])
            album_songs_req = requests.get(album_req.json()['tracks']['href'])
            songs = album_songs_req.json()['items']

            for song in songs:
                print 'Processing songs for album {}'.format(album_req.json()['name'].encode('ascii', 'ignore'))
                if song['preview_url']:
                    # Get song information (so we can get song popularity).
                    # Note that a list of genres is given in the album_req.
                    album_obj = create_album(album_req)
                    song_req = requests.get(song['href'])
                    song_obj = create_song(song_req, album_obj)
                    songs_for_year.append(song_obj)
                    songs_proc += 1
                    if songs_proc == num_songs:
                        return songs_for_year
                    # print 'Preview url found for song {}'.format(song_obj.name)
                    break

        req = requests.get(req.json()['albums']['next'])
        albums_proc += req.json()['albums']['limit']
        print ''

    return songs_for_year


# We'll use albums from the 60's to the 00's
years = range(1960, 1961)
songs_per_year = 2
print 'Getting {} songs from {}'.format(songs_per_year * len(years), years)
songs = []
for year in years:
    songs += get_songs(year, starting_offset, songs_per_year)


print 'Writing data to CSV...'
with open('song_data.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for song in songs:
        writer.writerow([song.song_id, song.name, song.year, song.popularity,
                         song.preview_url, song.binary_data])
