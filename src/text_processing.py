import re

def clean_text(text):
  text = text.lower()
  text = re.sub(r"http\S+","", text)       #Remove URLs
  text = re.sub(r"@\w+", "", text)         #Remove Mentions
  text = re.sub(r"[^a-z0-9\s]", "", text)  #Keep letters, numbers, spaces
  text = re.sub(r"\s+", " ", text)         #Normalize whitespace
  text = '<start> '+ text.strip() + ' <end>'
  return text


#Clean data from punctuation, unnecesary white space, and lower
def clean_data(imageCapDict):
  cleanedDict = {}
  for imageId in imageCapDict:
    cleanedDict[imageId] = []
    for caption in imageCapDict[imageId]:
      cleanedDict[imageId].append(clean_text(caption))
  return cleanedDict