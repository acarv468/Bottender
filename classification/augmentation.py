# Import Libraries
import os
import pandas as pd
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
import nltk
from datetime import datetime

os.environ["MODEL_DIR"] = '../model'

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Get most recent reviewed conversations
intent_df = pd.read_csv('data/reviewed_conversations/reviewed_conversation_18-Aug-2020(20-48-42).csv', encoding='utf-8-sig')

# Remove Nulls
intent_df = intent_df[pd.notnull(intent_df['intent'])]

print('Phrases to be augmented: ' + str(len(intent_df)))

####
# Augmentaion
####

# OCR Augmentation
ocr = nac.OcrAug()
ocr_list = []
for x in intent_df.iterrows():
    ocr_list.append(
        {
            'text': ocr.augment(x[1][0]),
            'intent': x[1][1]
        }
    )
ocr_aug_df = pd.DataFrame(ocr_list)
print(ocr_aug_df.head())

# Keyboard Augmentation
keyboard_aug = nac.KeyboardAug()
keyboard_list = []
for x in intent_df.iterrows():
    keyboard_list.append(
        {
            'text': keyboard_aug.augment(x[1][0]),
            'intent': x[1][1]
        }
    )
keyboard_aug_df = pd.DataFrame(keyboard_list)
print(keyboard_aug_df.head())

# Random Insert Augmentation
rand_insert = nac.RandomCharAug(action="insert")
rand_insert_list = []
for x in intent_df.iterrows():
    rand_insert_list.append(
        {
            'text': rand_insert.augment(x[1][0]),
            'intent': x[1][1]
        }
    )
rand_insert_df = pd.DataFrame(rand_insert_list)

# Random Substitue Augmentation
rand_sub = nac.RandomCharAug(action="substitute")
rand_sub_list = []
for x in intent_df.iterrows():
    rand_sub_list.append(
        {
            'text': rand_sub.augment(x[1][0]),
            'intent': x[1][1]
        }
    )
rand_sub_df = pd.DataFrame(rand_sub_list)

# Random Swap
rand_swap = nac.RandomCharAug(action="swap")
rand_swap_list = []
for x in intent_df.iterrows():
    rand_swap_list.append(
        {
            'text': rand_swap.augment(x[1][0]),
            'intent': x[1][1]
        }
    )
rand_swap_df = pd.DataFrame(rand_swap_list)

# Random Delete
rand_del = nac.RandomCharAug(action="delete")
rand_del_list = []
for x in intent_df.iterrows():
    rand_del_list.append(
        {
            'text': rand_del.augment(x[1][0]),
            'intent': x[1][1]
        }
    )
rand_del_df = pd.DataFrame(rand_del_list)

# Spelling Augmenter
spell_aug = naw.SpellingAug(dict_path='data/original_data/spelling_en.txt')
spell_aug_list = []
for x in intent_df.iterrows():
    spell_aug_list.append(
        {
            'text': spell_aug.augment(x[1][0]),
            'intent': x[1][1]
        }
    )
spell_aug_df = pd.DataFrame(spell_aug_list)

# Synonym Augmentation
syn_aug = naw.SynonymAug()
syn_aug_list = []
for x in intent_df.iterrows():
    syn_aug_list.append(
        {
            'text': syn_aug.augment(x[1][0]),
            'intent': x[1][1]
        }
    )
syn_aug_df = pd.DataFrame(syn_aug_list)

# Split Augmentation
split_aug = naw.SplitAug()
split_aug_list = []
for x in intent_df.iterrows():
    split_aug_list.append(
        {
            'text': split_aug.augment(x[1][0]),
            'intent': x[1][1]
        }
    )
split_aug_df = pd.DataFrame(split_aug_list)

####
# Concatenate DFs and export
####

combined_df = pd.concat([intent_df, ocr_aug_df, keyboard_aug_df, rand_insert_df, rand_sub_df, rand_swap_df, rand_del_df, spell_aug_df, syn_aug_df, split_aug_df])

# Remove duplicates
print('Number of phrases before removing duplicates: ' + str(len(combined_df)))
combined_df = combined_df.drop_duplicates(subset='text', keep='first')

print('Number of phrases after removing duplicates: ' + str(len(combined_df)))

datetime_obj = datetime.now()
timestamp_str = datetime_obj.strftime('%d-%b-%Y(%H-%M-%S)')
combined_df.to_csv('data/augmented_data/augmented_intents_' + timestamp_str +'.csv', index=False)
print('Augmented Intents saved as: augmented_intents_' + timestamp_str +'.csv')