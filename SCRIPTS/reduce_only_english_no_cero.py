# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:34:56 2024

@author: mfreschi
"""

import json
#from enchant.checker import SpellChecker
from langdetect import detect


max_error_count = 20
min_text_length = 2
number_reviews= 1000000

def is_in_english(quote):
  lang= 'it'
  try:
      lang = detect(quote)
  except:
      print('lang not recognised')
  return False if ((lang!= 'en') or len(quote.split()) < min_text_length) else True

# =============================================================================
# def is_in_english(quote):
#   d = SpellChecker("en_US")
#   d.set_text(quote)
#   errors = [err.word for err in d]
#   return False if ((len(errors) > max_error_count) or len(quote.split()) < min_text_length) else True
#
# =============================================================================



with open('./data_final/goodreads_reviews_dedup.json', 'r') as sourcefile, open('./data_final/reviews_english_no_cero.json', 'w') as destfile:
    intIndex=0
    for s in sourcefile:
        #if(intIndex==number_reviews):
        #     break;
        data = json.loads(s)
        #data.pop('book_id')
        data.pop('review_id')
        #data.pop('date_added')
        data.pop('date_updated')
        data.pop('read_at')
        data.pop('started_at')
        data.pop('n_votes')
        data.pop('n_comments')
        #check Language
        text_to_know=data['review_text']
        boolean_answer = is_in_english(text_to_know)
        if(boolean_answer):
            if data['rating']!=0 :
                json.dump(data, destfile)
                destfile.write('\n')
                intIndex+=1
    print("New database contains: ", str(intIndex), " entries")
