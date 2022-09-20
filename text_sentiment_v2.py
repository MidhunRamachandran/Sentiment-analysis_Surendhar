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

onfile = codecs.open("theme_classification_file.tsv", mode="w", encoding='utf-8')
onfile.write("UID\tProduct_ID\tReview_ID\tReview_Text\tProcessed_Text\tRating\tTextBlob\tVader\tAPI\tMatched_token\tTheme\n")

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

# theme_text = {"price":["pay","affordable","almighty dollar","amount","appraisal","appraisement","asking price","bank note","banknote","bankroll","barter","bill","bottom dollar","bounty","buck","bucks","capital","cash","charge","cheap","check","coin","coinage","compensation","cost","currency","disbursement","discount","dues","economical","estimate","expenditure","expense","expensive","fare","fee","figure","finances","folding money","fund","funds","gold","greenback","hard cash","inexpensive","legal tender","loot","medium of exchange","money","note","outlay","output","pay","payment","pesos","premium","price","price tag","prize","property","quotation","ransom","rate","reckoning","resonable","resources","retail","return","reward","riches","salary","setback","silver","single","specie","tariff","ticket","toll","top dollar","treasure","valuation","value","wad","wage","wages","wealth","wherewithal","wholesale","worth"],"shipping_delivery":["assignment","arrive","assortment","bag","baggage","batch","bottle","box","bunch","can","cargo","carrier","carting","carton","consignment","container","conveyance","crate","delayed","delivered","delivery","dispatch","distribution","drop","freight","freighting","giving over","goods","handing over","haul","impartment","intrusting","kit","load","lot","luggage","mailing","pack","package","packet","parcel","parcel post","pile","portage","post","prompt","relegation","rendition","sack","sailing","schedule","scheduled","sending shipment","sheaf","ship","shipment","speedy","stack","suitcase","surrender","tin","tracking","transmission","transmittal","transport","trunk","unit"],"size":["admeasurement","amount","amplitude","area","bigness","body","breadth","broadness","bulk","caliber","capaciousness","capacity","closeness","compactness","concentration","content","density","diameter","dimension","dimensions","enormity","extension","extent","fatness","greatness","heaviness","height","highness","hugeness","immensity","intensity","largeness","length","magnitude","mass","measurement","module","number","proportion","proportions","range","scope","sizable","solidity","spread","stature","stretch","substance","substantiality","tonnage","total","vast","vastness","volume","voluminosity","width"],"quality":["abiding","bad","adapt","aspect","attribute","backbone","blot","break","broke","bumpy","check","condition","constancy","constant","crack","defective","deficiency","deformity","dependable","description","dirt","diuturnal","drawback","dumbest","durable","durableness","element","endowment","endurance","enduring","erratic","error","essence","excellence","factor","failing","fast","fault","faulty","firm","fixed","flaw","flexibility","foible","frailty","gap","genius","glitch","grade","gremlin","grit","guarantee","guard","guts","gutsiness","hard as nails","imperishability","impervious","indestructible","individuality","infirmity","injury","intestinal fortitude","irregularity","issues","kind","lack","lasting","lastingness","long-continued","maintenance","make","mark","marring","merit","misrepresent","mistake","moxie","nature","parameter","patch","peculiarity","perdurable","perduring","perfect","permanence","permanent","persistence","persistent","polish","potential","predication","predisposes","problem","property","protection","quality","rank","recommend","reliable","repair","replace","resistant","return","rift","rough spot","savor","scarcity","scratch","second","shortage","shortcoming","speck","spot","squeak","stable","stain","stamina","standard","staying power","stick-to-itiveness","strong","stupid","sturdy","substantial","supportive","taint","tough","trait","unsoundness","vice","want","weak point","weakness","worth"],"operational_effectiveness":["act","accessible","action","activity","adaptable","adjust","affair","agency","application","attractive","baking","bigger","bit","boiling","brewing","broiling","browning","capability","carrying on","challenging","clout","cogency","compatible","confuse","connecting","consume","control","convenient","conveyance","course","custom","deal","deed","design","differential","directions","doing","easily operated","easy to understand","effect","efficacy","efficiency","effort","employment","engagement","enterprise","exercise","exercising","exertion","exploitation","failed","failure","feasible","feature","fit","flawlessly","flexible","foolproof","force","forcefulness","frying","function","grilling","handiwork","handy","happening","heating","height","in service","in working order","induction","influence","instrumentality","integration","labor","manageable","manipulation","message","motion","movement","noise","operate","operating","operative","performance","play","point","potency","power","practicable","practical","prepared","procedure","proceeding","process","program","progress","progression","ready","reliable","response","roasting","scheduling","sear","service","serviceable","simmering","simple","simplicity","sizzle","sizzling","steaming","steeping","stewing","straightforward","strength","success","texture","toasting","transaction","transference","uncomplicated","undertaking","untroublesome","usable","use","useful","validity","validness","verve","viable","vigor","wieldy","work","workable","working","workmanship"],"maintenance":["purification","ablution","aliment","alimentation","alimony","antisepsis","brushing","care","clean","conservation","continuance","deodorizing","disinfection","dusting","keep","nurture","preservertion","prolongation","prophylaxis","purgation","purge","regular","repair","sanitation","scouring","scrubbing","shampooing","sterilization","subsistance","support","sustaining","sweeping","tidying","upkeep","washing"]}
theme_text = {"price":['pay','affordable','almighty dollar','amount','appraisal','appraisement','asking price','bank note','banknote','bankroll','barter','bill','bottom dollar','bounty','buck','bucks','capital','cash','charge','cheap','coin','coinage','compensation','cost','currency','disbursement','discount','dues','economical','estimate','expenditure','expense','expensive','fare','fee','figure','finances','folding money','fund','funds','gold','greenback','hard cash','inexpensive','legal tender','loot','medium of exchange','money','note','outlay','output','payment','pesos','premium','price','price tag','prize','quotation','ransom','rate','reckoning','resonable','resources','retail','reward','riches','salary','setback','silver','single','specie','tariff','ticket','toll','top dollar','treasure','valuation','value','wad','wage','wages','wealth','wherewithal','wholesale'],"shipping_delivery":['assignment','arrive','assortment','bag','baggage','batch','bottle','box','bunch','can','cargo','carrier','carting','carton','consignment','container','conveyance','crate','delayed','delivered','delivery','dispatch','distribution','drop','freight','freighting','giving over','goods','handing over','haul','impartment','intrusting','kit','load','lot','luggage','mailing','pack','package','packet','parcel','parcel post','pile','portage','post','prompt','relegation','rendition','sack','sailing','schedule','scheduled','sending shipment','sheaf','ship','shipment','speedy','stack','suitcase','surrender','tin','tracking','transmission','transmittal','transport','trunk','unit','return'],"size":['admeasurement','amplitude','area','bigness','body','breadth','broadness','bulk','caliber','capaciousness','capacity','closeness','compactness','concentration','content','density','diameter','dimension','dimensions','enormity','extension','extent','fatness','greatness','heaviness','height','highness','hugeness','immensity','intensity','largeness','length','magnitude','mass','measurement','module','number','proportion','proportions','range','scope','sizable','solidity','spread','stature','stretch','substance','substantiality','tonnage','total','vast','vastness','volume','voluminosity','width'],"quality":['abiding','bad','adapt','aspect','attribute','backbone','blot','break','broke','bumpy','check','condition','constancy','constant','crack','defective','deficiency','deformity','dependable','description','dirt','diuturnal','drawback','dumbest','durable','durableness','element','endowment','endurance','enduring','erratic','error','essence','excellence','factor','failing','fast','fault','faulty','firm','fixed','flaw','flexibility','foible','frailty','gap','genius','glitch','grade','gremlin','grit','guarantee','guard','guts','gutsiness','hard as nails','imperishability','impervious','indestructible','individuality','infirmity','injury','intestinal fortitude','irregularity','issues','kind','lack','lasting','lastingness','long-continued','maintenance','make','mark','marring','merit','misrepresent','mistake','moxie','nature','parameter','patch','peculiarity','perdurable','perduring','perfect','permanence','permanent','persistence','persistent','polish','potential','predication','predisposes','problem','property','protection','quality','rank','recommend','reliable','replace','resistant','rift','rough spot','savor','scarcity','scratch','second','shortage','shortcoming','speck','spot','squeak','stable','stain','stamina','standard','staying power','stick-to-itiveness','strong','stupid','sturdy','substantial','supportive','taint','tough','trait','unsoundness','vice','want','weak point','weakness','worth'],"operational_effectiveness":['act','accessible','action','activity','adaptable','adjust','affair','agency','application','attractive','baking','bigger','bit','boiling','brewing','broiling','browning','capability','carrying on','challenging','clout','cogency','compatible','confuse','connecting','consume','control','convenient','course','custom','deal','deed','design','differential','directions','doing','easily operated','easy to understand','effect','efficacy','efficiency','effort','employment','engagement','enterprise','exercise','exercising','exertion','exploitation','failed','failure','feasible','feature','fit','flawlessly','flexible','foolproof','force','forcefulness','frying','function','grilling','handiwork','handy','happening','heating','in service','in working order','induction','influence','instrumentality','integration','labor','manageable','manipulation','message','motion','movement','noise','operate','operating','operative','performance','play','point','potency','power','practicable','practical','prepared','procedure','proceeding','process','program','progress','progression','ready','response','roasting','scheduling','sear','service','serviceable','simmering','simple','simplicity','sizzle','sizzling','steaming','steeping','stewing','straightforward','strength','success','texture','toasting','transaction','transference','uncomplicated','undertaking','untroublesome','usable','use','useful','validity','validness','verve','viable','vigor','wieldy','work','workable','working','workmanship'],"maintenance":['purification','ablution','aliment','alimentation','alimony','antisepsis','brushing','care','clean','conservation','continuance','deodorizing','disinfection','dusting','keep','nurture','preservertion','prolongation','prophylaxis','purgation','purge','regular','repair','sanitation','scouring','scrubbing','shampooing','sterilization','subsistance','support','sustaining','sweeping','tidying','upkeep','washing']}

##stemming
lem = WordNetLemmatizer()
theme_stem = {}
for x in theme_text:
	q = []
	for y in theme_text[x]:
		y = lem.lemmatize(y.strip(), pos="v")
		if y not in q:
			q.append(y)
		theme_stem[x] = q

df = pd.read_excel("Bigbasket Review.xlsx", encoding='utf-8', keep_default_na=False, na_values=[''], index=False)
df = df.fillna('')
counter = 0
json_data = json.loads(df.to_json(orient='index'))
for sku in json_data:
	review_id = str(json_data[sku]['UID'])
	product_id = str(json_data[sku]['Product_ID']).strip()
	review_title = str(json_data[sku]['Review_Title']).strip()
	review_text = str(json_data[sku]['Review_Text'])
	rating = str(json_data[sku]['Rating'])
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
			# onfile.write(str(counter) + "\t" + str(review_text) + "\t" + str(processed_text) + "\t" + str(j) + "\t" + polarity(pol_text_blob) + "\t" + polarity(pol_word_vader) + "\t" + str(pol_senti_api) + "\n")
			match_check = 0
			for m in theme_stem:
				for n in theme_stem[m]:
					if n in j:
						match_check = 1
						# print(n, m)
						counter += 1
						onfile.write(str(counter) + "\t" + str(product_id) + "\t" + str(review_id) + "\t" + str(review_text)
							+ "\t" + str(processed_text) + "\t" + str(rating) + "\t" + polarity(pol_text_blob) +
							"\t" + polarity(pol_word_vader) + "\t" + str(pol_senti_api) + "\t" + str(n) + "\t" + str(m) + "\n")
						print("Processed: "+ str(counter) + "\n")
			if match_check == 0:
				counter += 1
				onfile.write(str(counter) + "\t" + str(product_id) + "\t" + str(review_id) + "\t" + str(review_text)
                    + "\t" + str(processed_text) + "\t" + str(rating) + "\t" + polarity(pol_text_blob) + "\t" +
		            polarity(pol_word_vader) + "\t" + str(pol_senti_api) + "\tNo match\tNo match\n")
				print("Processed: "+ str(counter) + "\n")
		else:
			counter += 1
			pol_word_vader = pol_text_blob = pol_senti_api = processed_text = 'n/a'
			onfile.write(str(counter) + "\t" + str(product_id) + "\t" + str(review_id) + "\t" + str(review_text)
                + "\t" + str(processed_text) + "\t" + str(rating) + "\t" + polarity(pol_text_blob) + "\t" +
                polarity(pol_word_vader) + "\t" + str(pol_senti_api) + "\tNo match\tNo match\n")
			print("Processed: "+ str(counter) + "\n")
	else:
		counter += 1
		review_text = processed_text = j = pol_senti_api = pol_word_vader = pol_text_blob = 'n/a'
		onfile.write(str(counter) + "\t" + str(product_id) + "\t" + str(review_id) + "\t" + str(review_text)
            + "\t" + str(processed_text) + "\t" + str(rating) + "\t" + polarity(pol_text_blob) + "\t" +
            polarity(pol_word_vader) + "\t" + str(pol_senti_api) + "\tNo match\tNo match\n")
		print("Processed: "+ str(counter) + "\n")
