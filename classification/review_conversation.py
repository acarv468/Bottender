# Import Libraries
import pandas as pd
import glob
from datetime import datetime

# Get Latest Augmented Data
aug_intent_df = pd.read_csv('data/augmented_data/augmented_intents.csv', encoding='utf-8-sig')

# Get Conversation Log Data
path = r'data/conversation_logs'
all_files = glob.glob(path + "/*.csv")
li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
conv_log_df = pd.concat(li, axis=0, ignore_index=True)

####
# Process Conversation Logs
####

# Drop columns
conv_log_df = conv_log_df.drop(['Unnamed: 0', 'user_session', 'time', 'character', 'bot_response', 'predicted_intent_prob', 'is_toxic', 'is_toxic_prob', 'toxic', 'retrained_label'], axis=1)

# Remove duplicates
conv_log_df = conv_log_df.drop_duplicates(subset='user_input', keep='first')

# Finding unique inputs
conv_log_df = conv_log_df[~conv_log_df['user_input'].isin(aug_intent_df['text'])]

# Change Column Names
conv_log_df.rename(columns = {'user_input': 'text', 'predicted_intent':'intent'}, inplace=True)

print(conv_log_df)
print('Number of new inputs: ' + str(len(conv_log_df)))

####
# Export 
####

datetime_obj = datetime.now()
timestamp_str = datetime_obj.strftime('%d-%b-%Y(%H-%M-%S)')

# Export to reviewed conversations
conv_log_df.to_csv('data/reviewed_conversations/reviewed_conversation_' + timestamp_str + '.csv', index=False)
print('Reviewed conversation saved as: reviewed_conversation_ ' + timestamp_str + '.csv')
