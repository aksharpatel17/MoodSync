import json
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import boto3
from datetime import datetime

def lambda_handler(event, context):
    
    cilent_id = os.environ.get('client_id')
    client_secret = os.environ.get('client_secret')
    
    client_credentials_manager = SpotifyClientCredentials(client_id=cilent_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
    def get_first_playlist_ids_by_moods(moods):
        playlist_ids = {}
        
        for mood in moods:
            results = sp.search(q=f"mood:{mood}", type='playlist')
            
            if 'playlists' in results and 'items' in results['playlists']:
                first_playlist = results['playlists']['items'][0]
                playlist_id = first_playlist['id']
                playlist_ids[mood] = playlist_id
            else:
                playlist_ids[mood] = None
        
        return playlist_ids
    
    moods_to_search = ['happy', 'sad', 'focus', 'calm', 'angry']
    first_playlist_ids = get_first_playlist_ids_by_moods(moods_to_search)

    for mood, playlist_id in first_playlist_ids.items():
    
        
        spotify_data = sp.playlist_tracks(playlist_id)   
        
        cilent = boto3.client('s3')
        
        filename = mood + ".json"
        
        cilent.put_object(
            Bucket="mood-based-spotify-playlist",
            Key="raw_data/to_processed/" + filename,
            Body=json.dumps(spotify_data)
            )