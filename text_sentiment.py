import re, json, codecs, csv
import nltk
import requests
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import contractions

onfile = codecs.open("processed_file.tsv", mode="w", encoding='utf-8')
onfile.write("UID\tReview text\tProcessed text\tTokens\tTextBlob\tVader\tAPI\n")

def polarity(value):
	try:
		value = float(value)
		if value >= 0.05:
			pole = "POSITIVE"
		elif value < 0.05 and value > -0.05:
			pole = "NEUTRAL"
		elif value <= 0.05:
			pole = "NEGATIVE"
		else:
			pole = str(value)
	except:
		pole = "n/a"
	return pole


df = pd.read_excel("Bigbasket Review.xlsx", encoding='utf-8', keep_default_na=False, na_values=[''], index=False)
df = df.fillna('')
counter = 0
json_data = json.loads(df.to_json(orient='index'))
for sku in json_data:
	counter += 1
	review_title = str(json_data[sku]['Review_Title']).strip()
	review_text = str(json_data[sku]['Review_Text'])
	if review_title == 'n/a':
		review_title = ''
	if review_text == 'n/a':
		review_text = ''
	review_text = review_title + " " + review_text
	if review_text.strip():
		##lower case
		words = review_text.lower().strip()

		##cleaning
		words = re.sub(r'[^A-Za-z0–9\'\`]+', ' ', words)
		words = re.sub(r'\t+|\r+|\s+|\n+', ' ', words.strip())
		words = re.sub(r'\s+', ' ', words.strip())

		##tokenize
		tokens = nltk.word_tokenize(contractions.fix(words.strip()))
		# tokenizer = RegexpTokenizer(r'\w+')
		# tokens = tokenizer.tokenize(words)

		##stopwords removal
		##'can','will','do','did','could','would','has','have','had',
		stopwords = ['n/a','so','i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",'your','youre',"youve","youll",'youd','yours','yourself','yourselves','he','him','his','himself','she',"she's",'her ','shes','hers','herself','it',"it's",'its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that',"that'll",'these ','thatll','those','am','is','are','was','were','be','been','being','doing','a','an','the','and','if','or','because','as','until','while','of','at','by','for','with','about','between','into','through','during','before','after','to','from','in','out','on','off','then','once','here','there','when','where','why','how','all','any','both','each','will','ain','ma','shan',"shan't",'shan ','shant']
		# stopwords = [re.sub(r'[^A-Za-z0–9]+', ' ', sw.strip()) for sw in stopwords]
		print(words)
		words = [w.strip() for w in tokens if w.strip() not in stopwords]


		##stemming
		j = []
		k = ''
		for i in words:
			lem = WordNetLemmatizer()
			lem_word = lem.lemmatize(i.strip(), pos="v")
			k += lem_word.strip() + ' '
			j.append(lem_word)

		processed_text = k.strip()
		print(processed_text)
		if processed_text:
			pol_word_vader = sia.polarity_scores(k)['compound']
			# print("Vader polarity: " + str(pol_word_vader))
			pol_text_blob = TextBlob(processed_text).sentences[0].polarity
			# print("Textblob polarity: " + str(pol_text_blob))
			payload = "text=" + str(processed_text)
			try:
				response = requests.post('http://text-processing.com/api/sentiment/', data=payload, timeout=2)
				if response.status_code == 200:
					pol_senti_api = json.loads(response.text)['label']
				else:
					pol_senti_api = 'n/a'
			except Exception as e:
				print(e)
				pol_senti_api = str(e)
				pass
			if str(pol_senti_api) == "pos":
				pol_senti_api = "POSITIVE"
			if str(pol_senti_api) == "neg":
				pol_senti_api = "NEGATIVE"
			if str(pol_senti_api) == "neutral":
				pol_senti_api = "NEUTRAL"
			# print("Sentiment API polarity: " + pol_senti_api)
			onfile.write(str(counter) + "\t" + str(review_text) + "\t" + str(processed_text) + "\t" + str(j) + "\t" +
			            polarity(pol_text_blob) + "\t" + polarity(pol_word_vader) + "\t" + str(pol_senti_api) + "\n")
		else:
			pol_word_vader = pol_text_blob = pol_senti_api = processed_text = 'n/a'
			onfile.write(str(counter) + "\t" + str(review_text) + "\t" + str(processed_text) + "\t" + str(j) + "\t" +
			             polarity(pol_text_blob) + "\t" + polarity(pol_word_vader) + "\t" + str(pol_senti_api) + "\n")
	else:
		review_text = processed_text = j = pol_senti_api = pol_word_vader = pol_text_blob = 'n/a'
		onfile.write(str(counter) + "\t" + str(review_text) + "\t" + str(processed_text) + "\t" + str(j) + "\t" +
		             polarity(pol_text_blob) + "\t" + polarity(pol_word_vader) + "\t" + str(pol_senti_api) + "\n")