import sys
import io
import re
import json

dict = {}
with open('.\document.txt', 'r', encoding='utf-8') as f:
    i=0
    for line in f:
        word_text = re.split(r'([\W])+', line)
        word_text = [x for x in word_text if x not in [' ', '\n']]
        for w in word_text:
            if w not in dict: 
                dict[w]=1
            else:
                dict[w]+=1
        i+=1
        print(i)
f = open('dictionary.txt', 'w', encoding="utf-8")
f.write(json.dumps(dict))
