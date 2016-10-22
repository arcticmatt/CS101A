import sys

def fix_song_name(name):
    '''
    Eliminates non-ASCII characters from the passed-in
    string, returning the result
    '''
    pass

def decompose(line):
    '''
    Breaks down a CSV line into the portion before the song name, the song
    name, and the portion after the song name. Returns a tuple
    consisting of these three substrings.
    '''
    # Regex consisting of the portion of each line after the song name
    before_group = "([a-zA-Z0-9]+)"
    song_group = "(.*)"
    after_group = "([0-9]+,[0-9]+,https://.*)"
    pattern = "%s,%s,%s"%(before_group, song_group, after_group)
    match = re.match(pattern, line)
    return match.groups()

def fix_line(line):
    before, name, after = decompose(line)
    return "%s%s%s"%(before, fix_song_name(name), after)

def fix_scraped_csv(filename):
    f = open(filename)
    result_file = open("%s.fixed"%filename, "w+")
    for line in f:
        result_file.write("%s\n"%fix_line(line))
    result_file.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "usage: python fix_scraped_csv.py filename"
    fix_scraped_csv(sys.argv[1])
