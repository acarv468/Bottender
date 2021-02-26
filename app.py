import json
import time
import traceback
# import sqlalchemy
import pandas as pd
from numpy import random
from datetime import datetime
from predictions import *
from flask import Flask, render_template, request, Response, jsonify

app = Flask(__name__)

# Connect to the DB
# import mysql.connector
# import config
# mydb = mysql.connector.connect(
#     host= config.db_host,
#     user= config.db_user,
#     password= config.db_password,
#     database='conv_log_test_db'
# )
# mycursor = mydb.cursor()


 
# Load Initial State
response_df = pd.read_csv('state/response_data.csv')
state_tracking_df = pd.read_csv('state/state_tracking_with_tips.csv')
conversation_log_df = pd.read_csv('state/conversation_log_retrain.csv')
character = "Billy"
state = 'conversation'
session_number = 1

# Query db for new session number
# mycursor.execute('SELECT MAX(user_session) FROM conv_log_test')
# myresult = mycursor.fetchall()
# for x in myresult:
#     session_number = session_number + x[0]

# SQL Alchemy Engine
# database_connection = sqlalchemy.create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'.
#                                                 format(config.db_user, config.db_password, 
#                                                 config.db_host, 'conv_log_test_db'))

 
# @app.route("/")
# def home():
#     return render_template("index_reset2.html")

@app.route("/")
def intro():
    return render_template("intro.html")

@app.route("/game")
def game():
    print('You have reached the game')
    # print(state)
    # print(track.CHARACTER)
    return render_template("index.html")

#background process happening without any refreshing
@app.route('/reset')
def reset():
    global conversation_log_df
    global state_tracking_df
    global response_df
    global character
    global state
    global session_number
    # global mydb
    # global mycursor
    # global database_connection

    # # Upload conversation to DB
    # conversation_log_df.to_sql(con=database_connection, name='conv_log_test', if_exists='append')
   
    # Download csv of conversation log
    datetime_obj = datetime.now()
    timestamp_str = datetime_obj.strftime('%d-%b-%Y(%H-%M-%S)')
    print(timestamp_str)
    conversation_log_df.to_csv('classification/data/conversation_logs/conversation_' + timestamp_str + '.csv')

    response_df = pd.read_csv('state/response_data.csv')
    state_tracking_df = pd.read_csv('state/state_tracking_with_tips.csv')
    conversation_log_df = pd.read_csv('state/conversation_log_retrain.csv')
    character = "Billy"
    state = 'conversation'
    session_number = 1 + session_number

    # Query db for new session number
    # mycursor.execute('SELECT MAX(user_session) FROM conv_log_test')
    # myresult = mycursor.fetchall()
    # for x in myresult:
    #     session_number = session_number + x[0]
    
    print(session_number)
    print(state)

    print ("reset")
    return "nothing"

@app.route("/get")
def get_bot_response():
    user_submission = request.args.get('msg')

    global conversation_log_df
    global state_tracking_df
    global response_df
    global character
    global state
    global session_number

    print(state)

    # checking tags for what's already been said
    state_tags = list(state_tracking_df.loc[(state_tracking_df['times_predicted'] > 0), 'tag_name'])

    if state == 'conversation':
        # Predict intent
        predicted_intent = predict_intent(user_submission)[0]
        predicted_intent_prob = predict_intent(user_submission)[1]

        # Get Top 4 Intent Predictions
        top_intents_df = top_intents(user_submission)

        # Predict toxic
        toxic = ''
        is_toxic = predict_toxic(user_submission)[0]
        is_toxic_prob = predict_toxic(user_submission)[1]
        if (is_toxic == 1 and is_toxic_prob >= 0.9):
            toxic = 'yes'
        else:
            toxic = 'no'

        bot_response_four_tags = response_df[(response_df['intent'] == predicted_intent) 
                                            & (response_df['character'] == character) 
                                            & (response_df['toxic'] == toxic)
                                            & (response_df['tag_1'].isin(state_tags)) 
                                            & (response_df['tag_2'].isin(state_tags))
                                            & (response_df['tag_3'].isin(state_tags))
                                            & (response_df['tag_4'].isin(state_tags))]['response']
        bot_response_three_tags = response_df[(response_df['intent'] == predicted_intent) 
                                            & (response_df['character'] == character) 
                                            & (response_df['toxic'] == toxic)
                                            & (response_df['tag_1'].isin(state_tags)) 
                                            & (response_df['tag_2'].isin(state_tags))
                                            & (response_df['tag_3'].isin(state_tags))]['response']
        bot_response_two_tags = response_df[(response_df['intent'] == predicted_intent) 
                                            & (response_df['character'] == character) 
                                            & (response_df['toxic'] == toxic)
                                            & (response_df['tag_1'].isin(state_tags)) 
                                            & (response_df['tag_2'].isin(state_tags))]['response']
        bot_response_one_tag = response_df[(response_df['intent'] == predicted_intent) 
                                            & (response_df['character'] == character) 
                                            & (response_df['toxic'] == toxic)
                                            & (response_df['tag_1'].isin(state_tags))]['response']
        bot_response_no_tags = response_df[(response_df['intent'] == predicted_intent) 
                                            & (response_df['character'] == character) 
                                            & (response_df['toxic'] == toxic)
                                            & (response_df['tag_1'] == 'any')]['response']
        
        if len(bot_response_four_tags) > 0:
            bot_response = bot_response_four_tags
        elif len(bot_response_three_tags) > 0:
            bot_response = bot_response_three_tags
        elif len(bot_response_two_tags) > 0:
            bot_response = bot_response_two_tags
        elif len(bot_response_one_tag) > 0:
            bot_response = bot_response_one_tag
        else:
            bot_response = bot_response_no_tags

        # log conversation
        new_row = {'user_session': session_number, 'time': datetime.now(), 'character': character,'user_input': user_submission, 'bot_response': '', 
            'predicted_intent': predicted_intent, 'predicted_intent_prob': predicted_intent_prob, 'is_toxic': is_toxic, 'is_toxic_prob': is_toxic_prob, 'toxic': toxic, 'retrained_label': ''}

        # update tag df
        state_tracking_df.loc[((state_tracking_df['intent'] == predicted_intent) & (state_tracking_df['toxic'] == toxic)), 'times_predicted'] =\
        state_tracking_df.loc[((state_tracking_df['intent'] == predicted_intent) & (state_tracking_df['toxic'] == toxic)), 'times_predicted'] + 1

        # Tip calculation
        tip = max(state_tracking_df.loc[state_tracking_df['times_predicted'] > 0, 'value'].sum(), 0)

        # conversation flow
        if user_submission == 'bye':
            tip = tip + 0.5
            response = 'Bye! Stay safe. Your tip comes to $' + str(tip) + ". Thank you for helping to improve this project!"
            new_row['bot_response'] = 'Bye! Stay safe.'
            conversation_log_df = conversation_log_df.append(new_row, ignore_index=True)
            state = 'finished'
        elif(len(bot_response) > 0) & (predicted_intent_prob >= 0.95):
            response = random.choice(bot_response)
            new_row['bot_response'] = response
            conversation_log_df = conversation_log_df.append(new_row, ignore_index=True)
        elif predicted_intent_prob <= 0.95:
            response = "I'm sorry. I don't know what you're trying to say right now. But the more you talk to me the more I learn!"
            new_row['bot_response'] = response
            conversation_log_df = conversation_log_df.append(new_row, ignore_index=True)
        elif len(bot_response) == 0:
            response = "I know what you're trying to say but I don't have a programmed response for it."
            new_row['bot_response'] = response
            conversation_log_df = conversation_log_df.append(new_row, ignore_index=True)
        else:
            response = "you've reached the end of the road."

        #conversation_log_df = conversation_log_df.append(new_row, ignore_index=True)
        print('Session number is: '+ str(session_number))
        print(conversation_log_df.head())
        print('Length of conversation is: ' + str(len(conversation_log_df)))         
    elif state == 'finished':
        response = 'You have finished the game. Please click the reset button to try again. Thank you!'
    else:
        response = 'You are not in any state.'

    return response

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    if intent_model:
        try:
            json_ = request.json
            user_submission = json_['user_submission']
            print(user_submission)
            # query = pd.get_dummies(pd.DataFrame(json_))
            # query = query.reindex(columns=model_columns, fill_value=0)

            intent_prediction = predict_intent(user_submission)
            toxic_prediction = predict_toxic(user_submission)

            return jsonify({'intent_prediction': str(intent_prediction), 'toxic_prediction': str(toxic_prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
 
if __name__ == "__main__": 
	app.run(host ='0.0.0.0', port = 5001, debug = True)