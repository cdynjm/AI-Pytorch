import json
import os

intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": [
                "Hi", "Hello", "Hey", "Howdy", "hey nigga",
                "Good morning", "Good afternoon", "Good evening", "What's up", "Yo"
            ],
            "responses": ["Hello!", "Hi there!", "Hey!", "Greetings!", "Howdy!"]
        },
        {
            "tag": "goodbye",
            "patterns": [
                "Bye", "See you", "Goodbye", "See you later",
                "Farewell", "Catch you later", "Take care"
            ],
            "responses": ["Bye!", "See you later!", "Goodbye!", "Take care!", "Have a great day!"]
        },
        {
            "tag": "thanks",
            "patterns": [
                "Thanks", "Thank you", "Much appreciated",
                "Thanks a lot", "Thanks so much", "Thank you very much"
            ],
            "responses": ["You're welcome!", "No problem!", "Anytime!", "My pleasure!", "Glad to help!"]
        },
        {
            "tag": "help",
            "patterns": ["Help", "I need help", "Can you help me?", "Support", "Assist me", "I have a problem"],
            "responses": ["Sure, how can I assist you?", "I'm here to help! What do you need?", "Please tell me how I can support you."]
        },
        {
            "tag": "hours",
            "patterns": ["What are your hours?", "When are you open?", "Opening hours", "Business hours", "Working hours"],
            "responses": ["We are open Monday to Friday from 9am to 6pm.", "Our working hours are 9am - 6pm, Monday through Friday."]
        },
        {
            "tag": "location",
            "patterns": ["Where are you located?", "Location", "Address", "Where can I find you?", "Office location"],
            "responses": ["We are located at 123 Main Street, Cityville.", "Our office is at 123 Main Street, Cityville."]
        },
        {
            "tag": "payments",
            "patterns": ["What payment methods do you accept?", "Payments", "How can I pay?", "Payment options"],
            "responses": ["We accept credit cards, PayPal, and bank transfers.", "You can pay using credit cards, PayPal, or bank transfers."]
        },
        {
            "tag": "complaint",
            "patterns": ["I want to complain", "Complaint", "I'm not happy", "Issue", "Problem with service"],
            "responses": ["I'm sorry to hear that. Please tell me more about the issue.", "We apologize for the inconvenience. How can we fix this?"]
        },
        {
            "tag": "jokes",
            "patterns": ["Tell me a joke", "Make me laugh", "Funny", "Joke please", "Say something funny"],
            "responses": ["Why don't scientists trust atoms? Because they make up everything!", "I told my computer I needed a break, and now it won’t stop sending me Kit-Kat ads!"]
        }
    ]
}

# ---------- Path to intents.json ----------
filename = "intents.json"

# ---------- Overwrite the file ----------
with open(filename, "w", encoding="utf-8") as f:
    json.dump(intents, f, ensure_ascii=False, indent=2)

print(f"[✔] intents.json has been (re)generated with {len(intents['intents'])} intents.")
