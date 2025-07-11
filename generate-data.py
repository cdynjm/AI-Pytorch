import json

intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": [
                "Hi", "Hello", "Hey", "Howdy", "hey nigga",
                "Good morning", "Good afternoon", "Good evening",
                "What's up", "Yo", "Hi there, how are you doing?",
                "Hey there, good to see you!", "Hello, I hope you're having a nice day!",
                "Hey, what's going on today?", "Good evening, I wanted to say hi."
            ],
            "responses": [
                "Hello!", "Hi there!", "Hey!", "Greetings!", "Howdy!"
            ]
        },
        {
            "tag": "goodbye",
            "patterns": [
                "Bye", "See you", "Goodbye", "See you later", "Farewell",
                "Catch you later", "Take care",
                "Thanks for your help, bye!", "Alright, goodbye for now.",
                "Talk to you later, take care.", "I have to go now, see you next time!"
            ],
            "responses": [
                "Bye!", "See you later!", "Goodbye!", "Take care!", "Have a great day!"
            ]
        },
        {
            "tag": "thanks",
            "patterns": [
                "Thanks", "Thank you", "Much appreciated", "Thanks a lot",
                "Thanks so much", "Thank you very much",
                "I appreciate your help a lot.", "Thank you for assisting me today.",
                "Thanks for being so helpful.", "Thanks, that was exactly what I needed."
            ],
            "responses": [
                "You're welcome!", "No problem!", "Anytime!", "My pleasure!", "Glad to help!"
            ]
        },
        {
            "tag": "help",
            "patterns": [
                "Help", "I need help", "Can you help me?", "Support", "Assist me",
                "I have a problem", "Can I get some help with this issue?",
                "I’m having trouble with something.", "Could you please assist me?",
                "I really need some support right now."
            ],
            "responses": [
                "Sure, how can I assist you?",
                "I'm here to help! What do you need?",
                "Please tell me how I can support you."
            ]
        },
        {
            "tag": "hours",
            "patterns": [
                "What are your hours?", "When are you open?", "Opening hours",
                "Business hours", "Working hours",
                "Could you tell me when you're open?", "When can I visit your office?",
                "What time do you open and close?", "I’d like to know your business schedule."
            ],
            "responses": [
                "We are open Monday to Friday from 9am to 6pm.",
                "Our working hours are 9am - 6pm, Monday through Friday."
            ]
        },
        {
            "tag": "location",
            "patterns": [
                "Where are you located?", "Location", "Address", "Where can I find you?",
                "Office location",
                "Can you tell me where your office is?", "Where is your company based?",
                "What’s your exact address?", "I want to visit your office, where is it?"
            ],
            "responses": [
                "We are located at 123 Main Street, Cityville.",
                "Our office is at 123 Main Street, Cityville."
            ]
        },
        {
            "tag": "payments",
            "patterns": [
                "What payment methods do you accept?", "Payments", "How can I pay?",
                "Payment options",
                "Do you accept credit cards?", "Can I pay via PayPal?",
                "Is bank transfer available for payment?", "What are my options to settle the bill?"
            ],
            "responses": [
                "We accept credit cards, PayPal, and bank transfers.",
                "You can pay using credit cards, PayPal, or bank transfers."
            ]
        },
        {
            "tag": "complaint",
            "patterns": [
                "I want to complain", "Complaint", "I'm not happy", "Issue", "Problem with service",
                "I have a complaint about your service.", "I’m dissatisfied with how things went.",
                "There’s something wrong I’d like to report.", "I need to file a complaint."
            ],
            "responses": [
                "I'm sorry to hear that. Please tell me more about the issue.",
                "We apologize for the inconvenience. How can we fix this?"
            ]
        },
        {
            "tag": "jokes",
            "patterns": [
                "Tell me a joke", "Make me laugh", "Funny", "Joke please", "Say something funny",
                "Do you know any good jokes?", "Can you make me laugh a bit?",
                "Say something to cheer me up.", "Got a joke for me?"
            ],
            "responses": [
                "Why don't scientists trust atoms? Because they make up everything!",
                "I told my computer I needed a break, and now it won’t stop sending me Kit-Kat ads!"
            ]
        }
    ]
}

# Save the file
with open("intents.json", "w", encoding="utf-8") as f:
    json.dump(intents, f, ensure_ascii=False, indent=2)

print(f"[✔] intents.json has been regenerated with {len(intents['intents'])} rich intents.")