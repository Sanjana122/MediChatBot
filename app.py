# Import necessary libraries and modules
import re
import spacy 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Define the DialogueState class
class DialogueState:
    def __init__(self):
        self.current_goal = None
        self.identified_entities = {}
        self.previous_utterances = []

    def set_current_goal(self, goal):
        self.current_goal = goal

    def add_identified_entity(self, entity_type, entity_value):
        self.identified_entities[entity_type] = entity_value

    def add_user_utterance(self, utterance):
        self.previous_utterances.append({"speaker": "User", "text": utterance})

    def add_chatbot_utterance(self, utterance):
        self.previous_utterances.append({"speaker": "Chatbot", "text": utterance})
        self.current_goal = utterance  # Set the current goal as the latest chatbot response

    def add_previous_utterance(self, utterance):
        self.previous_utterances.append(utterance)

    def get_current_goal(self):
        return self.current_goal

    def get_identified_entities(self):
        return self.identified_entities

    def get_previous_utterances(self):
        return self.previous_utterances

# Define the DialogueFSM class
class DialogueFSM:
    def __init__(self):
        # Define the states of the FSM.
        self.states = {
            "start": StartState(),
            "transition_to_next_state": TransitionToNextState(),
            "identify_disease": IdentifyDiseaseState(),
            "predict_symptoms": PredictSymptomsState(),
            "provide_medication": ProvideMedicationState(),
            "end": EndState()
        }

        # Set the current state.
        self.current_state = "start"
        self.dialogue_state = DialogueState()

    def transition(self, input_utterance):
        # Get the next state based on the current state and the input utterance.
        next_state = self.states[self.current_state].get_next_state(input_utterance)

        # Set the current state to the next state.
        self.current_state = next_state

        # Execute actions for the current state.
        self.states[self.current_state].execute_actions(input_utterance, self.dialogue_state)

# Define the StartState class
class StartState:
    def get_next_state(self, input_utterance):
        # Always transition to an intermediate state before moving to the actual state.
        return "transition_to_next_state"

    def execute_actions(self, input_utterance, dialogue_state):
        # Greet the user and provide instructions on how to use the system.
        response = "Welcome to the medical chatbot. I can help you find information about symptoms, diseases, and medications. To get started, please tell me about your symptoms."
        print(response)
        dialogue_state.add_chatbot_utterance(response)

# Define the TransitionToNextState class
class TransitionToNextState:
    def get_next_state(self, input_utterance):
        # Add logic here to determine the next state based on the input.
        # For example, if "symptom" is in the input, transition to "identify_disease".
        if "symptom" in input_utterance:
            return "diabetes"
        # If the input doesn't contain symptoms and it's not a specific command, transition to "start".
        elif not any(cmd in input_utterance.lower() for cmd in ["medication", "disease"]):
            return "start"
        # Add more conditions as needed.
        else:
            return "transition_to_next_state"

    def execute_actions(self, input_utterance, dialogue_state):
        # No actions needed for this state.
        pass

# ... (previous code)

# Define the IdentifyDiseaseState class
class IdentifyDiseaseState:
    def get_next_state(self, input_utterance):
        # Load your dataset
        dataset = pd.read_csv("data/tokenized_dataset.csv")  # Update with your actual dataset path
        print(dataset)

        # Extract symptoms from the user's input
        symptoms = extract_symptoms(input_utterance)

        # Check if any symptoms are identified
        if symptoms:
            # Iterate through the dataset to find the disease with the highest symptom match
            max_match = 0
            identified_disease = "unknown"
            
            for index, row in dataset.iterrows():
                dataset_symptoms = eval(row['symptoms'])  # Convert string representation of list to actual list
                match_count = sum(symptom in dataset_symptoms for symptom in symptoms)
                
                if match_count > max_match:
                    max_match = match_count
                    identified_disease = row['disease']

            # Set a match threshold (adjust as needed)
            match_threshold = 2  # Example threshold, you can adjust based on your dataset
            if max_match >= match_threshold:
                # Add the identified disease and symptoms to the dialogue state
                dialogue_state.add_identified_entity("disease", identified_disease)
                dialogue_state.add_identified_entity("symptoms", symptoms)

                # Switch to the next state to predict symptoms
                return "predict_symptoms"

        # Set a default response if no disease is identified
        response = "I'm sorry, but I couldn't identify a specific disease based on the provided symptoms. Please consult with a healthcare professional for a more accurate diagnosis."

        # Add the user's input and chatbot's response to the dialogue state
        dialogue_state.add_user_utterance(input_utterance)
        dialogue_state.add_chatbot_utterance(response)

        # Print the response to the user
        print(response)

# ... (remaining code)


# Define the PredictSymptomsState class
class PredictSymptomsState:
    def get_next_state(self, input_utterance):
        # For simplicity, transition to the provide_medication state directly
        return "Insulin, Metphormin"

    def execute_actions(self, input_utterance, dialogue_state):
        # Placeholder logic for predicting symptoms
        identified_disease = dialogue_state.get_identified_entities().get("disease", "unknown disease")

        # Replace this with your actual logic for predicting symptoms based on the identified disease
        predicted_symptoms = predict_symptoms(identified_disease)

        # Add the predicted symptoms to the dialogue state
        dialogue_state.add_identified_entity("predicted_symptoms", predicted_symptoms)

        # Generate a response based on the predicted symptoms
        response = f"I predict that you may also experience the following symptoms: {', '.join(predicted_symptoms)}."

        # Add the chatbot's response to the dialogue state
        dialogue_state.add_chatbot_utterance(response)

        # Print the response to the user
        print(response)

# Placeholder function for predicting symptoms (replace with your actual implementation)
def predict_symptoms(identified_disease):
    # Replace this with your actual logic for predicting symptoms based on the identified disease
    # For now, return a dummy list of symptoms
    return ["weight loss", "fatigue", "restlessness"]

# Define the ProvideMedicationState class
class ProvideMedicationState:
    def get_next_state(self, input_utterance):
        # Stay in the provide_medication state.
        return "provide_medication"

    def execute_actions(self, input_utterance, dialogue_state):
        # Get the identified disease from the dialogue state.
        identified_disease = dialogue_state.get_identified_entities().get("disease", "unknown disease")

        # Lookup medication for the disease (you might have a more sophisticated lookup mechanism).
        medication_for_disease = lookup_medication(identified_disease)

        # Generate a response with medication information.
        medication_response = f"The medication for {identified_disease} is: {medication_for_disease}"

        # Add the medication response to the dialogue state.
        dialogue_state.add_chatbot_utterance(medication_response)

        # Print the medication response to the user.
        print(medication_response)

# Define the EndState class
class EndState:
    def get_next_state(self, input_utterance):
        # Stay in the end state.
        return "end"

    def execute_actions(self, input_utterance, dialogue_state):
        # Provide a closing message.
        response = "Thank you for using the medical chatbot. If you have more questions, feel free to ask."
        print(response)
        dialogue_state.add_chatbot_utterance(response)
        dialogue_state.set_current_goal(response)

# Add a new function to extract symptoms from user input using spaCy
def extract_symptoms(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    symptoms = [ent.text for ent in doc.ents if ent.label_ == "SYMPTOM"]
    return symptoms if symptoms else ["No symptoms identified"]

# ... (other code)

# Your Flask app logic goes here
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="templates")

# Instantiate the DialogueFSM
dialog_fsm = DialogueFSM()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        
        # Process the user's input
        dialog_fsm.transition(user_input)

        # Get the current state of the dialogue
        current_state = dialog_fsm.current_state

        # Execute actions based on the current state
        if current_state == "identify_disease":
            execute_identify_disease_actions(dialog_fsm.dialogue_state)
        elif current_state == "predict_symptoms":
            execute_predict_symptoms_actions(dialog_fsm.dialogue_state)
        elif current_state == "provide_medication":
            execute_provide_medication_actions(dialog_fsm.dialogue_state)
        elif current_state == "end":
            execute_end_actions(dialog_fsm.dialogue_state)
        
        # Get the chatbot's response from the dialogue state
        chatbot_response = dialog_fsm.dialogue_state.get_current_goal()

        # Return the response as HTML
        return jsonify({"response": chatbot_response})
    
    # Render the chatbot interface
    return render_template("index.html", dialogue_state=dialog_fsm.dialogue_state)

def execute_identify_disease_actions(dialogue_state):
    # Get the identified disease and symptoms from the dialogue state
    identified_disease = dialogue_state.get_identified_entities().get("disease", "unknown disease")
    symptoms = dialogue_state.get_identified_entities().get("symptoms", [])

    # Perform actions based on the identified disease and symptoms
    if identified_disease != "unknown disease":
        # Do something with the identified disease (e.g., log it, store it in a database, etc.)
        print(f"Identified Disease: {identified_disease}")

        # If there are identified symptoms, you can also perform actions with them
        if symptoms:
            print(f"Identified Symptoms: {', '.join(symptoms)}")
            # Perform additional actions with the identified symptoms

    else:
        # Handle the case where no disease is identified
        print("No specific disease identified. Consider consulting a healthcare professional.")


def execute_predict_symptoms_actions(dialogue_state):
    # Get the identified disease and predicted symptoms from the dialogue state
    identified_disease = dialogue_state.get_identified_entities().get("disease", "unknown disease")
    predicted_symptoms = dialogue_state.get_identified_entities().get("predicted_symptoms", [])

    # Perform actions based on the identified disease and predicted symptoms
    if identified_disease != "unknown disease":
        # Do something with the identified disease (e.g., log it, store it in a database, etc.)
        print(f"Identified Disease: {identified_disease}")

        # If there are predicted symptoms, you can also perform actions with them
        if predicted_symptoms:
            print(f"Predicted Symptoms: {', '.join(predicted_symptoms)}")
            # Perform additional actions with the predicted symptoms

    else:
        # Handle the case where no disease is identified
        print("No specific disease identified. Consider consulting a healthcare professional.")


def execute_provide_medication_actions(dialogue_state):
    # Get the identified disease and medication information from the dialogue state
    identified_disease = dialogue_state.get_identified_entities().get("disease", "unknown disease")
    medication_for_disease = lookup_medication(identified_disease)

    # Perform actions based on the identified disease and medication information
    if identified_disease != "unknown disease":
        # Do something with the identified disease (e.g., log it, store it in a database, etc.)
        print(f"Identified Disease: {identified_disease}")

        # If there is medication information, you can also perform actions with it
        if medication_for_disease:
            print(f"Medication for {identified_disease}: {medication_for_disease}")
            # Perform additional actions with the medication information

    else:
        # Handle the case where no disease is identified
        print("No specific disease identified. Consider consulting a healthcare professional.")

# Function to lookup medication for a given disease (replace with your actual implementation)
def lookup_medication(identified_disease):
    # Replace this with your actual logic for looking up medication based on the identified disease
    # For now, return a dummy medication
    return "Dummy Medication for " + identified_disease


def execute_end_actions(dialogue_state):
    # Perform actions for the "end" state
    print("End state reached. Performing final actions...")

    # You might want to perform any cleanup or finalization steps here

    # For example, you could log the entire conversation history
    print("Conversation History:")
    for utterance in dialogue_state.get_previous_utterances():
        print(f"{utterance['speaker']}: {utterance['text']}")

    # You can also perform any other specific actions needed for ending the conversation

    # Optionally, you can log or store the final state of the dialogue state
    final_state = {
        "current_goal": dialogue_state.get_current_goal(),
        "identified_entities": dialogue_state.get_identified_entities(),
        "previous_utterances": dialogue_state.get_previous_utterances(),
    }
    print("Final Dialogue State:", final_state)

    # You can add more actions based on your specific requirements


# ... (remaining code)

if __name__ == "__main__":
    app.run(debug=True)
