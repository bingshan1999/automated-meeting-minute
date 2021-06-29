import nltk
import string
import spacy
import numpy as np
import settings
import collections

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def download_package():
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('averaged_perceptron_tagger')
  nltk.download('wordnet')
  nltk.download('vader_lexicon')
download_package()

# define stopword list
nltk_stopwords = stopwords.words('english')
custom_stopwords = ['hmm', 'um', 'uh-huh', 'okay', 'uh', 'yeah', 'mm-hmm', 'uhm', 'mm', 'yep', 'thanks','sure', 
                    'maybe', 'great', 'shall', 'nope','okay', 'alright', 'would', 'could', 'sorry', 'oh', "'s", "'m", "'re", 'hi',
                    'think']
filter_word = nltk_stopwords + custom_stopwords
embed = spacy.load("en_core_web_md")

def euclidean(x,y):
  distance = np.sqrt(np.sum((x - y) ** 2))
  sim = 1/(1+distance)
  return sim

# Group sentences to their topics by keyword matching
def group_topic(topic_sentence, sentence_list, name):
  for sentence in sentence_list:
    for topic in topic_sentence:
      topic_bi = topic.split()
      if any(x in sentence[1] for x in topic_bi):
        if name in topic_sentence[topic]:
          topic_sentence[topic][name].append(sentence) 
        else:
          topic_sentence[topic][name] = [sentence]
        break
  return topic_sentence

# Write dictionary to file, separated by speaker
def write_dict_to_file(path, dictionary):
  f = open(path, "w")
  
  for topic in dictionary:
    count = 0
    num_speaker = dictionary[topic].keys()
    for i in num_speaker:
      count += len(dictionary[topic][i])
    
    if count > 2:
      f.write("Topic {} \n".format(topic))

      for name in dictionary[topic]:
        f.write("Person " + name + ": \n")
        for (line_num,sentence) in dictionary[topic][name]:
          f.write(sentence + " ")
        f.write("\n")

      f.write("\n\n")
  f.close()
  print("File {} created.".format(path))
  return

# Filter the transcript to find all the nouns and keeping sentences that have noun present
def POS_filter(transcript):
  document = nltk.sent_tokenize(transcript)
  
  POS_sentence = [] #used to store all sentence that have noun present
  NOUNS_list = [] #used to store all nouns present

  for i, sentence in enumerate(document):
    if len(sentence) > 2: 
      tokens = word_tokenize(sentence)
      pos = nltk.pos_tag(tokens)
      noun_present = False 
      index = -1
      noun_present_in_sent = []

      for j, part in enumerate(pos):
        if part[1] in ('NN','NNP') and part[0].lower() not in filter_word and len(list(part[0])) > 1:
          noun_present = True 
          # check if previous term is also noun to create bigram
          if index == -1:        
            noun_present_in_sent.append(part[0].lower())
          elif index != -1 and j-1 == index: #previous term is a noun 
              noun_present_in_sent[-1] = noun_present_in_sent[-1] + " " + part[0].lower()
          else:
            if part[0].lower() not in noun_present_in_sent:       
              noun_present_in_sent.append(part[0].lower())
          
          index = j

      if noun_present:
        POS_sentence.append(tuple((i,sentence)))
        NOUNS_list.extend(noun_present_in_sent)
  #print("NUM OF SEN: {}, NUM OF SENTENCE WITH NOUN: {}".format(len(document),len(POS_sentence)))
  return NOUNS_list, POS_sentence

# Used to calculate line number difference and semantic similarity
def cal_relevance(dictionary, person, noise):
  for topic in dictionary:
    diff = 0
    prev = 0
    counter = 0
    prev_embedding = []
    total_distance = 0

    if person in dictionary[topic]:
      for (line_num,sentence) in dictionary[topic][person]:
        # Calculate similarity of two sentences
        current_embedding = embed(' '.join(stem(sentence)))
        if not prev_embedding:
          prev_embedding = current_embedding
        else:
          distance = euclidean(prev_embedding.vector,current_embedding.vector) 
          total_distance += distance
          prev_embedding = current_embedding
        
        # Calculate line number difference of two sentence
        if counter == 0:
          prev = line_num
        else:
          diff += (line_num - prev)
          prev = line_num
          
        counter += 1
      
      if counter -1 == 0:
        #print("One sentence only")
        continue
      else:
        #print("Topic: {}, avg semantic difference: {}, avg line difference: {}".format(topic,total_distance/(counter-1),diff/(counter-1)))
        if total_distance/(counter-1) < 0.26 or diff/(counter-1) > 20:
          #print(topic)
          noise.append(topic)

  return dictionary, noise


# For performance measure
def compare(x_token, y_token):
  y_token_copy = y_token
  count = len(y_token)

  for x in x_token:
    if x in y_token:
      y_token_copy.remove(x)
  
  count_copy = len(y_token_copy)
  if (count - count_copy) >1:
    return True, y_token_copy
  else: 
    return False, y_token

# Function to calculate the accuracy of the transcript
def performance_measure(sum_path, minute_path, transcript_path):
  Y1 = open(minute_path, 'r')
  Y2 = open(sum_path, 'r')
  Y3 = open(transcript_path, 'r')

  predicted_sentence = nltk.sent_tokenize(Y1.read())
  sentence_y = nltk.sent_tokenize(Y2.read())
  sentence_x = nltk.sent_tokenize(Y3.read())
  ground_truth = []

  #print("Document: {}, X: {}, Y: {}".format(document,len(sentence_x),len(sentence_y)))
  y_count = 0
  count = 0
  for index_x,value_x in enumerate(sentence_x):
    x_token = word_tokenize(value_x)
    y_token = word_tokenize(sentence_y[y_count])
    is_decision, y_token = compare(x_token,y_token)
    if is_decision:
      ground_truth.append(value_x)

    if not y_token:
      y_count += 1
  
  for sent in predicted_sentence.copy():
    if sent.startswith('Topic') or sent.startswith('Person'):
      predicted_sentence.remove(sent)

    elif sent in ground_truth:
      count += 1
  
  print("length of transcript: {}, length of summary: {}, len of extracted sen: {}, len of minute:{}".format(len(sentence_x),len(sentence_y),len(ground_truth),len(predicted_sentence)))
  return 

#read document
documents = {
  'A' : open(settings.transcript_A, 'r'),
  'B' : open(settings.transcript_B, 'r'),
  'C' : open(settings.transcript_C, 'r'),
  'D' : open(settings.transcript_D, 'r'),
}

document_pos = {}
topic_sentence = {}


for document in documents:
  doc = documents[document].read()
  noun, pos= POS_filter(doc)
  index_to_pop = []
  
  document_pos[document] = pos

  # Look for repetitive noun
  for i, n in enumerate(noun): 
    neigbour_index = [i+x for x in range(1,settings.num_neighbour+1)]
    neigbour = [noun[index] for index in neigbour_index if index > 0 and index < len(noun)]
    if n not in neigbour:
      index_to_pop.append(i) #save the index of nouns that does not present in neighbour 
  
  # Reverse the index list to avoid messing with the index while popping
  index_to_pop.reverse()
  for i in index_to_pop:
    noun.pop(i)

  # Save the important noun in a nested dictionary
  for n in noun:
    if n in topic_sentence:
      continue
    else:
      topic_sentence[n] = {}
  
#print(topic_sentence)
noise = []

for document in document_pos:
  # Loop per speaker to match the filtered sentence to the topic
  topic_sentence = group_topic(topic_sentence, document_pos[document], document)
  if settings.topic_filter:
    topic_sentence, noise = cal_relevance(topic_sentence, document, noise)

if settings.topic_filter:
  dup_noise = [item for item, count in collections.Counter(noise).items() if count > 1]
  for topic in topic_sentence.copy():
    if len(topic_sentence.copy()[topic].items()) > 1 and topic in dup_noise:
      topic_sentence.pop(topic)
    elif len(topic_sentence.copy()[topic].items()) == 1 and topic in noise:
      topic_sentence.pop(topic)

write_dict_to_file(settings.output_path, topic_sentence)
performance_measure(settings.extsum_path, settings.output_path, settings.full_transcript)



