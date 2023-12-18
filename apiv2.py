import requests
import datetime
import requests
import datetime
from vertexai.preview import generative_models
from vertexai.preview.generative_models import GenerativeModel

from google.cloud import aiplatform
from google.oauth2 import service_account
import json

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

# Initialize the Vertex AI SDK
aiplatform.init(
    project='dsports-6ab79',
    location='us-central1',  # change this if you're using a different region
    staging_bucket='gs://artifacts.dsports-6ab79.appspot.com',
    credentials=credentials,
)


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