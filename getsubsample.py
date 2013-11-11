import random,nltk,string

docs = []
test_docs = []
def init_subsample():
  global docs
  f = open('./ALL/documents.dat', 'r')
  for line in f:
    line = line.strip()
    if line:
      docs.append(line)

def init_test_data(test_num):
  global test_docs
  global docs
  test_docs = [ docs[i] for i in random.sample(xrange(len(docs)), test_num) ]
  for i in test_docs:
    docs.remove(i)

def get_test_docs():
  global test_docs
  return test_docs

def get_subsample(n):
  global docs
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

