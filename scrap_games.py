import json
import requests


def get_game(game_id):
    r = requests.post(
    	'https://www.codingame.com/services/gameResultRemoteService/findByGameId',
    	json = [str(game_id), None]
    )
    replay = r.json()
    for frame in replay['frames'][1:]:
        frame.pop('view')
    with open(f'game_logs/{game_id}.json', 'w+') as f:
    	f.write(json.dumps(replay, indent=4))


game_ids_bot_beam_160_03_16 = [
   877959870,
   877959752,
   877977892,
   877960780,
]

for game_id in game_ids_bot_beam_160_03_16:
    get_game(game_id)