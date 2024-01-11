import json
import boto3


def lambda_handler(event, context):
    s3 = boto3.client('s3')

    bucket_name = "mood-based-spotify-playlist"
    mood = event.get("data")

    key = f"raw_data/to_processed/{mood}.json"

    
    response = s3.get_object(Bucket=bucket_name, Key=key)
    results = json.loads(response['Body'].read().decode('utf-8'))
    
    tracks_info = []

    
    for track in results['items']:
        track_info = {
            'id': track['track']['id'],
            'track_name': track['track']['name'],
            'artist': track['track']['artists'][0]['name'],
            'album_cover_url': track['track']['album']['images'][0]['url'],
            'track_url': track['track']['external_urls']['spotify'],
            'album_name': track['track']['album']['name']
        }
    tracks_info.append(track_info)
        

    return {
        'statusCode': 200,
        'body': json.dumps({'data': tracks_info})
    }
