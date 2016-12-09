"""
Script for processing existing scraped data to eliminate non-alphanumeric
characters from song names.
"""
import sys
import re

def fix_song_name(name):
    '''
    Returns a copy of the passed-in string with non-alphanumeric chars
    removed. For empty strings, returns None (to filter out songs consisting
    entirely of non-alphanumeric chars, which are likely non-English songs).
    '''
    result = ''.join(ch for ch in name if ch.isalnum() or ch == ' ')
    if len(result) > 0:
        return result
    return None

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
    fixed_name = fix_song_name(name)
    if fixed_name is not None:
        return "%s,%s,%s"%(before, fixed_name, after)
    return None

def fix_scraped_csv(filename):
    f = open(filename)
    result_file = open("%s.fixed"%filename, "w+")
    for line in f:
        processed_line = fix_line(line)
        if processed_line is not None:
            result_file.write("%s\n"%fix_line(line))
    result_file.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "usage: python fix_scraped_csv.py filename"
    fix_scraped_csv(sys.argv[1])
