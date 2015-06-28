

import re
import requests
import sqlite3
import sys
import pandas as pd
from bs4 import BeautifulSoup
from nltk import word_tokenize, pos_tag
from dateutil.parser import parse


def get_obama_speeches():
    # Start with pages that list the radio addresses.
    initial_urls = ["https://www.whitehouse.gov/briefing-room/weekly-address?page="
                    + str(i) for i in range(33)]

    # Obtain links to pages hosting transcripts for individual radio addresses.
    # Also create lists of titles and dates matching these links.
    transcript_urls = []
    titles = []
    dates = []
    base_url = "https://www.whitehouse.gov"
    for url in initial_urls: 
        soup = BeautifulSoup(requests.get(url).text, "lxml")
        div_of_links = soup("div", {"class": "view-content"})[-1]      
        a_tags = [a_tag for a_tag in div_of_links("a", href=True)]
        transcript_urls += [base_url + a_tag["href"] for a_tag in a_tags]
        long_titles = [a_tag.get_text() for a_tag in a_tags]
        titles += [re.sub("weekly address:", "", long_title, flags=re.IGNORECASE) 
                   for long_title in long_titles]
        dates += [span.get_text() for span in div_of_links("span")]  

    # Download text from transcripts.
    speeches = []
    transcript_class = {"class": "field-name-field-transcript"}
    for url in transcript_urls:
        speech_soup = BeautifulSoup(requests.get(url).text, "lxml")
        transcript_divs = [div for div in speech_soup("div", transcript_class)] 
        content = transcript_divs[0].get_text()
        speeches.append(content) 

    return speeches, titles, dates

def get_bush_speeches():
    # Start with the single page that lists all the radio addresses.
    initial_url = "http://georgewbush-whitehouse.archives.gov/news/radio/"

    # Obtain links to pages hosting transcripts for individual radio addresses. 
    soup = BeautifulSoup(requests.get(initial_url).text, "html5lib")  
    table_of_links = soup("table", {"class": "archive"})[0]
    base_url = "http://georgewbush-whitehouse.archives.gov"  
    transcript_urls = [base_url + a_tag["href"] 
                       for a_tag in table_of_links("a", href=True)]

    # Obtain dates and titles matching transcript links.
    trs = table_of_links("tr")
    dates = []
    titles = []
    for tr in trs:
        tds = tr("td")
        if len(tds) == 1: 
            year = tds[0].get_text()
        elif len(tds) == 2: 
            dates.append(tds[0].get_text() + year)
            titles.append(tds[1].get_text())

    # Download text from transcripts.
    speeches = []
    for url in transcript_urls: 
        speech_soup = BeautifulSoup(requests.get(url).text, "lxml")
        transcript_divs = speech_soup("div", {"id": "news_container"})
        try:
            div_with_paragraphs = transcript_divs[0]
        except IndexError:
            speeches.append(None)
            print "Failed to locate transcript from the following url:", url
            continue
        paragraphs = []
        for tag in div_with_paragraphs.contents:
            if tag.name == "p":
                paragraphs.append(tag.get_text())
            elif tag.name is None:
                # This only occurs when the tag is actually just text. 
                paragraphs.append(unicode(tag)) 
        speeches.append(paragraphs)

    return speeches, titles, dates
     
def translate_from_unicode(text):
    unicode_vocab = {"-": [u"\u2013"],
                     "--": [u"\u2014", u"\u2015"], 
                     "'": [u"\u2018", u"\u2019", u"\u201b", u"\u201c", 
                           u"\u201d", u"\u2032"],
                     ",": [u"\u201a"],  
                     "/": [u"\u2044"],
                     "...": [u"\u2026"], 
                     " ": [u"\xa0", u"\t", u"\r"], 
                     "n": [u"\xf1"]}
    for char, codes in unicode_vocab.items():
        for code in codes:
            text = text.replace(code, char)
    return str(text)

def remove_extra_spaces(text):
    return " ".join(text.split())

def extract_obama_speech(content):
    # Immediately return None if content is None or the empty string.
    if not content:
        return None

    # Start by removing footer information and replacing unicode symbols.
    chunk = translate_from_unicode(content.split("# ")[0])

    # Return None if radio address was given by Michelle Obama or Joe Biden.
    wrong_speakers = ["THE FIRST LADY", "Michelle Obama", "Joe Biden"]
    if [speaker for speaker in wrong_speakers if speaker in chunk]:
        return None

    # Split into lines to make it possible to extract only Obama's speech.
    # Also remove extra whitespace because it's no longer needed after splitting.
    lines = [remove_extra_spaces(line) 
             for line in chunk.splitlines() 
             if re.search("[a-zA-Z]", line)]

    # Skip intro text by finding the first line of the speech header.
    i = 0
    while i < len(lines) and "Remarks" not in lines[i]:
        i += 1
        
    # Skip the speech header by finding the next line that has enough 
    #   words to really be a paragraph.
    i += 1
    while i < len(lines) and len(lines[i].split()) < 10:
        i += 1
    
    # Return None if the resulting index is out of range.
    if i >= len(lines):
        return None

    # Return Obama's speech as a single string.
    return "\n\n".join(lines[i: ])

def extract_bush_speech(paragraphs):
    # Immediately return None if no paragraphs are provided.
    if not paragraphs:
        return None

    # Start by replacing unicode symbols and removing extra whitespace.
    lines = [remove_extra_spaces(translate_from_unicode(p)) 
             for p in paragraphs 
             if re.search("[a-zA-Z]", p)]

    # Combine all lines that occur after Bush begins speaking.
    try:
        # Bush is usually introduced in one of two ways.
        merged = "\n\n".join(lines)
        chunk = re.split("PRESIDENT: *|BUSH: *", merged)[1]
    except IndexError:
        # If Bush isn't introduced nicely, skip the header by starting
        #   at the first line that has enough words to really be a paragraph.
        i = 0
        while len(lines[i].split()) < 15:
            i += 1
        chunk = "\n\n".join(lines[i:])

    # Remove all text that occurs after Bush stops speaking.
    chunk = re.split("[^\.\!\?]*END", chunk)[0]

    # Return None if radio address was given by Laura Bush.
    if "Laura Bush" in chunk:
        return None
    
    # Return Bush's speech as a single string.
    return chunk

# Translates each word in a speech to its part of speech.
def translate_to_pos(speech):
    if not speech:
        return None
    pos_pairs = pos_tag(word_tokenize(speech))
    return " ".join([pair[1] for pair in pos_pairs])

def extract_title(text):
    return remove_extra_spaces(translate_from_unicode(text))

def extract_date(text):
    dt = parse(translate_from_unicode(text))
    return str(dt.date())

def download_process_store(database_name):
    print "Downloading Obama's radio addresses..."
    raw_obama_speeches, raw_obama_titles, raw_obama_dates = get_obama_speeches()

    print "Downloading Bush's radio addresses..."
    raw_bush_speeches, raw_bush_titles, raw_bush_dates = get_bush_speeches()

    print "Extracting and cleaning... "
    obama_speeches = [extract_obama_speech(raw_speech)
                      for raw_speech in raw_obama_speeches]
    bush_speeches = [extract_bush_speech(raw_speech) 
                      for raw_speech in raw_bush_speeches]                    
    speeches = obama_speeches + bush_speeches
    pos = [translate_to_pos(speech) for speech in speeches]
    titles = [extract_title(raw_title) 
              for raw_title in raw_obama_titles + raw_bush_titles]
    dates = [extract_date(raw_date) for raw_date in raw_obama_dates + raw_bush_dates]
    speakers = ["obama"]*len(obama_speeches) + ["bush"]*len(bush_speeches)

    print "Storing..."
    addresses = pd.DataFrame({"speech": speeches,
                              "pos": pos, 
                              "title": titles, 
                              "date": dates, 
                              "speaker": speakers})
    addresses = addresses.drop_duplicates(subset="speech").dropna(axis=0)
    addresses["id"] = range(addresses.shape[0])
    con = sqlite3.connect(database_name)
    addresses.to_sql("radio_addresses", con, index=False, if_exists="replace")

if __name__ == '__main__':
    try: 
        database_name = sys.argv[1]
    except IndexError:
        print "usage: make_data.py database_name"
        sys.exit("\nExecution failed: user must provide location for sql database.")
    download_process_store(database_name)
    
        

