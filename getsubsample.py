import random,nltk,string

docs = []
def init_subsample():
  #stop_words = [v.strip() for v in file('stop_words.txt').readlines()]
  f = open('ap.txt', 'r')
  for line in f:
    if line[0] == '<' or line[1] == '<':
      continue
    else:
      docs.append(line)

def get_subsample(n):
  rand_smpl = [ docs[i] for i in random.sample(xrange(len(docs)), n) ]
  return rand_smpl

def cleanup():
  f = open('ap_noformat.dat','r')
  for line in f:
    line.translate(string.maketrans(string.punctuation, ' '*len(string.punctuation)))
    words = line.split()
    print line
    print words
    raw_input()

