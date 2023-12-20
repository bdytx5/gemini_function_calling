import requests
import datetime
import requests
import datetime
from vertexai.preview import generative_models
from vertexai.preview.generative_models import GenerativeModel
import wandb
from google.cloud import aiplatform
from google.oauth2 import service_account
import json
import time 
# docs 
# https://cloud.google.com/vertex-ai/docs/start/install-sdk
# https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/function-calling
# Base API URLs for each sport


cred = {
    "type": "service_account",
    "project_id": "[Your Google Cloud project ID]",
    "private_key_id": "[Unique identifier for the private key]",
    "private_key": "-----BEGIN PRIVATE KEY-----\n[Your private key here]\n-----END PRIVATE KEY-----",
    "client_email": "[Service account email address]",
    "client_id": "[Service account client ID]",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "[URL to the service account's public x509 certificate]"
}


credentials = service_account.Credentials.from_service_account_info(cred)

aiplatform.init(
    project='your project id',
    location='region. eg: us-central1',  # change this if you're using a different region
    staging_bucket='your storage bucket',
    credentials=credentials,
)
# Initialize the Vertex AI SDK


api_urls = {
    "football": "http://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
    "basketball": "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "baseball": "http://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
}


leagues = {
    "football": "nfl",
    "basketball": "nba",
    "baseball": "mlb"
}

def get_events(sport):
    """ Get all event IDs for a given sport """
    url = api_urls.get(sport)
    if not url:
        return "Sport not supported"

    response = requests.get(url)
    if response.status_code != 200:
        return "Failed to fetch data"

    events = response.json().get('events', [])
    event_ids = [event['id'] for event in events]
    return event_ids

def get_event_info(sport, all_events_in_sport, teamName):
    """ Get description of each event given a list of its IDs and sport """
    event_info_dict = {}
    for event_id in all_events_in_sport:
        # Constructing the URL based on the sport and event ID
        url = f"https://sports.core.api.espn.com/v2/sports/{sport}/leagues/{leagues[sport]}/events/{event_id}"

        response = requests.get(url)
        if response.status_code != 200:
            event_info_dict[event_id] = "Failed to fetch event data"
            continue

        event_data = response.json()

        # Extracting the name of the event which includes team names
        event_name = event_data.get('name', 'Unknown Event')

        # Extracting the date and time of the event
        date_str = event_data.get('date', '')
        if date_str:
            date_str = date_str.replace('Z', '+00:00')
            game_datetime = datetime.datetime.fromisoformat(date_str)
            formatted_time = game_datetime.strftime("%B %d, %Y at %I:%M %p UTC")
        else:
            formatted_time = "Time not available"

        event_info_dict[event_id] = {
            "description": f"{event_name}", 
            "time": f"{formatted_time}", 
        }
    for ev_id in event_info_dict.keys():
        if teamName.lower() in event_info_dict[ev_id]['description'].lower():
            return event_info_dict[ev_id]['time']

    return {}



def getGameTime(sport, teamName):    
    event_ids = get_events(sport)
    event_info = get_event_info(sport, event_ids, teamName)
    return event_info

# gemini needs to figure out which sport api to use given the query eg call getGameTime properly 

q = "what time do the chiefs play?"
## use gemini function calling to determine the params for the following function 

# Initialize the Gemini model
model = GenerativeModel("gemini-pro")
    


get_game_time_func = generative_models.FunctionDeclaration(
    name="get_game_time_from_query",
    description="Determine the game time from a query, given a sport. Supported sports are football, basketball, and baseball.",
    parameters={
        "type": "object",
        "properties": {
            "teamName": {
                "type": "string",
                "description": "The name of one of the teams mentioned in the query, for example 'chiefs' or 'cardinals'"
            },
            "sport": {
                "type": "string",
                "enum": ["football", "basketball", "baseball"],
                "description": "The sport to search in. "
            }
        },
        "required": ["query", "sport"]
    },
)




gametime_tool = generative_models.Tool(
    function_declarations=[get_game_time_func]
)

# Example query
q = "what time do the chiefs play?"

# Use Gemini model to determine the sport from the query -> gets the params (sport and team name)
model_response = model.generate_content(
    q,
    generation_config={"temperature": 0},
    tools=[gametime_tool],
)
args = model_response.candidates[0].content.parts[0].function_call.args.pb


print(getGameTime(args.get("sport").string_value, args.get("teamName").string_value))


###### W&B eval 

# Initialize wandb
wandb.init(project="Game Time Analysis")

# Create a wandb Table to log the results
results_table = wandb.Table(columns=["Query", "Predicted Sport", "Predicted Team", "Ground Truth Sport", "Ground Truth Team"])

queries_and_truths = [
    # Football queries
    ("what time do the chiefs play?", ("football", "Chiefs")),
    ("when is the next packers game?", ("football", "Packers")),
    ("time for the broncos game", ("football", "Broncos")),
    ("dallas cowboys game schedule", ("football", "Cowboys")),
    ("new england patriots next game", ("football", "Patriots")),
    ("seahawks game start time", ("football", "Seahawks")),
    ("buccaneers upcoming game", ("football", "Buccaneers")),
    ("49ers game tonight", ("football", "49ers")),
    
    # Basketball queries
    ("lakers game tonight?", ("basketball", "Lakers")),
    ("heat game start time", ("basketball", "Heat")),
    ("warriors next game time", ("basketball", "Warriors")),
    ("celtics game schedule", ("basketball", "Celtics")),
    ("bucks next game", ("basketball", "Bucks")),
    ("suns game time", ("basketball", "Suns")),
    
    # Baseball queries
    ("yankees game today", ("baseball", "Yankees")),
    ("dodgers game start time", ("baseball", "Dodgers")),
    ("time for braves next game", ("baseball", "Braves")),
    ("mets game schedule", ("baseball", "Mets")),
    ("cubs game tonight", ("baseball", "Cubs")),
    ("astros next game", ("baseball", "Astros"))
]


# Function to check accuracy
def check_accuracy(predicted, truth):
    return 1 if predicted.lower() == truth.lower() else 0

# Process each query
correct_sports, correct_teams, total_queries = 0, 0, len(queries_and_truths)
for query, (truth_sport, truth_team) in queries_and_truths:
    # Call the model for inference
    model_response = model.generate_content(
        query,
        generation_config={"temperature": 0},
        tools=[gametime_tool],
    )
    args = model_response.candidates[0].content.parts[0].function_call.args.pb
    predicted_sport = args.get("sport").string_value
    predicted_team = args.get("teamName").string_value

    # Calculate accuracies
    sport_accuracy = check_accuracy(predicted_sport, truth_sport)
    team_accuracy = check_accuracy(predicted_team, truth_team)
    correct_sports += sport_accuracy
    correct_teams += team_accuracy

    # Log to wandb table
    results_table.add_data(query, predicted_sport, predicted_team, truth_sport, truth_team)
    time.sleep(10)


wandb.log({f"Game_Time_Queries": results_table})
# Calculate and log overall accuracies
overall_sport_accuracy = correct_sports / total_queries
overall_team_accuracy = correct_teams / total_queries
wandb.log({"Overall Sport Accuracy": overall_sport_accuracy, "Overall Team Accuracy": overall_team_accuracy})

