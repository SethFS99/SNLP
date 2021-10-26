# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 23:37:16 2020

@author: Seth
"""
import lyricsgenius as genius
import csv,os

def get_lyrics(names, songs,sentiment, genius,file):
    c = 0
    badSongs = []#create list of songs that may not work as intended
    for i in range(len(names)):
        try:
            song = genius.search_song(songs[i], names[i])
            try:#sentiement is dont in <<|*|>> format for easy parsing later on
            
                file.write("\n \n"+song.lyrics+ "\n \n<<"+sentiment[i]+">>")#occasionally songs aren't compatable, not sure why as they should be
            except:
                badSongs.append(songs[i])
                print("Song could not be written continuing...")
                continue
            c += 1
        except:
            print(f"some exception at {names[i]}: {c}")
    print(f"Successfully wrote {c} songs")
    return badSongs
#Putting these here in case i need them later          
#clientKey -bp0IsCDALrBknI2R0KdQH0xOxPPpGavQ8DjB5_AvKYBSJ1HqDk6KmDtI3eiRHNVCzppPcEBfiMKHhCzIzY4_A
#ClientID nIRWj-vzT73_y9NkzuK0dryvwIT49PNhpGlRXJxyWGSxWgFfS-bbV-VRgnqeu1g-
    # Insert access token
genius = genius.Genius('eOo8fgIk3HyopTIZ6NeAnj_M24xL3ms_N7PKaRXeYhTmxR0M3rFsr0yrszQ3a95g', skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True)
file = open(os.getcwd()+"/TestLyrics.txt", "w")
artists=[]
sentiment=[]
songs=[]
with open('TestSet.csv', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        
        sentiment.append(row[7])
        artists.append(row[6])
        songs.append(row[2])
artists.pop(0)
songs.pop(0)
bad_songs=get_lyrics(artists,songs, sentiment, genius, file)
file.close()