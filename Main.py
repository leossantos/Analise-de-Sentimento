from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
#Leitura
arq = open('amazon_cells_labelled.txt', 'r')
amazon = []
amazonC = []
for linha in arq:
    frase = linha.replace('\t', ' ').replace('\n', '')
    amazonC.append(frase[len(frase)-1])
    frase = frase[0:len(frase)-2]
    amazon.append(frase)
print(amazon)
print(amazonC)
arq.close()
arq = open('imdb_labelled.txt', 'r')
imdb = []
imdbC = []
for linha in arq:
    frase = linha.replace('\t', ' ').replace('\n', '')
    imdbC.append(frase[len(frase)-1])
    frase = frase[0:len(frase) - 2]
    imdb.append(frase)
print(imdb)
print(imdbC)
arq.close()
arq = open('yelp_labelled.txt', 'r')
yelp = []
yelpC = []
for linha in arq:
    frase = linha.replace('\t', ' ').replace('\n', '')
    yelpC.append(frase[len(frase)-1])
    frase = frase[0:len(frase) - 2]
    yelp.append(frase)
print(yelp)
print(yelpC)
##################################################
#Fazendo Bag of Words
vectorizer = CountVectorizer(analyzer="word")
freq_imdb = vectorizer.fit_transform(imdb)
###############################
#Criando o modelo Byeas e fitando os dados
modelo = MultinomialNB()
modelo.fit(freq_imdb, imdbC)
#################
#Testando acuracia com base de dados Amazon
freq_amazon = vectorizer.transform(amazon)
resulAmazon = modelo.predict(freq_amazon)
print(metrics.accuracy_score(amazonC, resulAmazon))
####################################
#Testando a acuracia com base de dados yelp
freq_yelp = vectorizer.transform(yelp)
resulYelp = modelo.predict(freq_yelp)
print(round(metrics.accuracy_score(yelpC, resulYelp)*100))
#############################################
#Unindo os 3 bancos
frases = amazon + imdb + yelp
classes = amazonC + imdbC + yelpC
print(frases)
print(classes)
#####################################
#Fazendo Cross Validation
freq = vectorizer.fit_transform(frases)
modelo = MultinomialNB()
resultados = cross_val_predict(modelo, freq, classes, cv=10)
print(metrics.accuracy_score(classes, resultados))
################################
v2 = CountVectorizer(ngram_range=(1,4))
freq = v2.fit_transform(frases)
modelo = MultinomialNB()
resultados = cross_val_predict(modelo, freq, classes, cv=10)
print(metrics.accuracy_score(classes, resultados))
sen = ['0', '1']
print(metrics.classification_report(classes, resultados, sen))
