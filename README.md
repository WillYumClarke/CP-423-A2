CP 423 Text Retrieval and Search Engines
Assignment 2
William Clarke 190524800
Andrew Best 190620060

Methodology:
The provided programs follows a methodology for the information retrieval using the TF-IDF technique
The program starts by loading and preprocessing a dataset, which contains multiple docs.
The dataset is tokenizied, converted to lowercase, stop words removed, and puncuation eliminated.
The positional index data structure is then developed which holds the position of every word in every doc.
Next the TF-IDF is calculated, the matrix represents the importance of each term in 
each doc but using TF within the doc and within the corpus. It does this for every weighting scheme aside
from double normalization due to how muc longer the program takes to execute when 249 documents are in the dataset.
The program then prompts the user to enter a phrase query which is then processed in the same
preprocess_text function used to process the data. It then uses the search function to search for
the top 5 relevant documents based on their TF-IDF scores, it calculates the query vector and matces
it with the document vectors in the matrix to find the docs with the highest scores.
The program then uses the TF-IDF to calculate the cosine similarity between the query vector and doc vectors.
it returns the top 5 docs based on cosine similarity.

Double Normalization:
the double normalization significanlty slows down the execution of the program. To test the program with
double normalization simply uncomment the following lines:
66, 84, 130-138, 148-149, 204-209

Preprocessing Steps:
1) Convert text to lowercase
	text = text.lower()
2) Tokenize the  text
	tokens = word_tokenize(text)
3) Remove stopwords
	stop_words = set(stopwords.words('english'))
	tokens = [token for token in tokens if token not in stop_words]
4) Remove punctuation marks and empty spaces
	tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]
	tokens = [token for token in tokens if token != '']
5) Return tokens
	return tokens

Outputs:
-------------------------------------------------------------------------------------------
Enter the phrase query: dogs

Top 5 relevant documents based on TF-IDF score with binary term frequency weighting scheme:
psf.txt DocID: 190, Score: 0.07807076753407109
lpeargrl.txt DocID: 154, Score: 0.0635072209956009
mtinder.txt DocID: 169, Score: 0.0630417447073773
13chil.txt DocID: 2, Score: 0.04662138922669604
mcdonaldl.txt DocID: 160, Score: 0.04130732121536556
Total docs returned: 27

Top 5 relevant documents based on TF-IDF score with raw count term frequency weighting scheme:     
mtinder.txt Doc ID: 169, Score: 0.09832303521498593
bulhuntr.txt Doc ID: 50, Score: 0.0503202620462989
13chil.txt Doc ID: 2, Score: 0.04786494485431673
arctic.txt Doc ID: 30, Score: 0.028614028128516062
aesop11.txt Doc ID: 20, Score: 0.02771875211880637
Total docs returned: 27

Top 5 relevant documents based on TF-IDF score with term frequency term frequency weighting scheme:
mtinder.txt Doc ID: 169, Score: 0.09832298313775599
bulhuntr.txt Doc ID: 50, Score: 0.05032021172241617
13chil.txt Doc ID: 2, Score: 0.047864923610959696
arctic.txt Doc ID: 30, Score: 0.02861399669770499
aesop11.txt Doc ID: 20, Score: 0.02771870421081082
Total docs returned: 27

Top 5 relevant documents based on TF-IDF score with log normalization term frequency weighting scheme:
mtinder.txt Doc ID: 169, Score: 0.11280713114734733
13chil.txt Doc ID: 2, Score: 0.07391583094211288
bulhuntr.txt Doc ID: 50, Score: 0.05463279744957157
psf.txt Doc ID: 190, Score: 0.051461946059101975
lpeargrl.txt Doc ID: 154, Score: 0.045198162196351456
Total docs returned: 27

Top 5 relevant documents based on cosine similarity and binary weighting scheme:
psf.txt Doc ID: 190, Score: 0.07807076781350336
lpeargrl.txt Doc ID: 154, Score: 0.06350722118050474
mtinder.txt Doc ID: 169, Score: 0.06304174488958057
13chil.txt Doc ID: 2, Score: 0.04662138932634438
mcdonaldl.txt Doc ID: 160, Score: 0.04130732129359206
Total docs returned: 27

Top 5 relevant documents based on cosine similarity and raw count weighting scheme:
mtinder.txtDoc ID: 169, Score: 0.0983230353257886
bulhuntr.txtDoc ID: 50, Score: 0.05032026207532076
13chil.txtDoc ID: 2, Score: 0.04786494488057554
arctic.txtDoc ID: 30, Score: 0.028614028141028314
aesop11.txtDoc ID: 20, Score: 0.02771875212132241
Total docs returned: 27

Top 5 relevant documents based on cosine similarity and term frequency weighting scheme:
mtinder.txtDoc ID: 169, Score: 0.09832303532578864
bulhuntr.txtDoc ID: 50, Score: 0.050320262075320754
13chil.txtDoc ID: 2, Score: 0.04786494488057553
arctic.txtDoc ID: 30, Score: 0.028614028141028328
aesop11.txtDoc ID: 20, Score: 0.02771875212132252
Total docs returned: 27

Top 5 relevant documents based on cosine similarity and log normalization weighting scheme:
mtinder.txt Doc ID: 169, Score: 0.11280713150983941
13chil.txt Doc ID: 2, Score: 0.07391583109774538
bulhuntr.txt Doc ID: 50, Score: 0.054632797534593755
psf.txt Doc ID: 190, Score: 0.0514619462342667
lpeargrl.txt Doc ID: 154, Score: 0.045198162331470254
Total docs returned: 27

-------------------------------------------------------------------------------------------

Enter the phrase query: computer science

Top 5 relevant documents based on TF-IDF score with binary term frequency weighting scheme:
life.txt DocID: 147, Score: 0.058871286492628704
mydream.txt DocID: 172, Score: 0.05129134359145295
stairdre.txt DocID: 218, Score: 0.04077716143608995
emperor3.txt DocID: 83, Score: 0.03900738396579377
plescopm.txt DocID: 183, Score: 0.030316386731769975
Total docs returned: 53

Top 5 relevant documents based on TF-IDF score with raw count term frequency weighting scheme:
bulphrek.txt Doc ID: 58, Score: 0.06520934021818907
stairdre.txt Doc ID: 218, Score: 0.06275849876399645
life.txt Doc ID: 147, Score: 0.05397837993098789
mindprob.txt Doc ID: 163, Score: 0.05086677707181357
bulfelis.txt Doc ID: 49, Score: 0.03465025753511628
Total docs returned: 53

Top 5 relevant documents based on TF-IDF score with term frequency term frequency weighting scheme:
bulphrek.txt Doc ID: 58, Score: 0.06520929218435595
stairdre.txt Doc ID: 218, Score: 0.06275844790059981
life.txt Doc ID: 147, Score: 0.05397836142373218
mindprob.txt Doc ID: 163, Score: 0.05086675191467213
bulfelis.txt Doc ID: 49, Score: 0.03465022933779902
Total docs returned: 53

Top 5 relevant documents based on TF-IDF score with log normalization term frequency weighting scheme:
stairdre.txt Doc ID: 218, Score: 0.057254894927158656
life.txt Doc ID: 147, Score: 0.056493855449991526
bulphrek.txt Doc ID: 58, Score: 0.05116785852666033
mindprob.txt Doc ID: 163, Score: 0.04497692275460348
dskool.txt Doc ID: 80, Score: 0.04221877958534759
Total docs returned: 53

Top 5 relevant documents based on cosine similarity and binary weighting scheme:
life.txt Doc ID: 147, Score: 0.06436028370585072
mydream.txt Doc ID: 172, Score: 0.056073607691963914
stairdre.txt Doc ID: 218, Score: 0.040640092161627964
emperor3.txt Doc ID: 83, Score: 0.03616608190921081
discocanbefun.txt Doc ID: 78, Score: 0.031562105166239765
Total docs returned: 53

Top 5 relevant documents based on cosine similarity and raw count weighting scheme:
bulphrek.txtDoc ID: 58, Score: 0.07073955866196577
stairdre.txtDoc ID: 218, Score: 0.06020277948107691
life.txtDoc ID: 147, Score: 0.059011175938118776
mindprob.txtDoc ID: 163, Score: 0.055609455543194686
bulfelis.txtDoc ID: 49, Score: 0.03788095231045959
Total docs returned: 53

Top 5 relevant documents based on cosine similarity and term frequency weighting scheme:
bulphrek.txtDoc ID: 58, Score: 0.07073955866196574
stairdre.txtDoc ID: 218, Score: 0.060202779481076926
life.txtDoc ID: 147, Score: 0.05901117593811879
mindprob.txtDoc ID: 163, Score: 0.05560945554319468
bulfelis.txtDoc ID: 49, Score: 0.037880952310459584
Total docs returned: 53

Top 5 relevant documents based on cosine similarity and log normalization weighting scheme:
life.txt Doc ID: 147, Score: 0.061761187633152484
stairdre.txt Doc ID: 218, Score: 0.05559953605674149
bulphrek.txt Doc ID: 58, Score: 0.05401600606049228
mindprob.txt Doc ID: 163, Score: 0.04917044745654836
mydream.txt Doc ID: 172, Score: 0.04317497475911904
Total docs returned: 53

-------------------------------------------------------------------------------------------

Enter the phrase query: people dog train boat plane car
Query length must be less than or equal to than 5.

-------------------------------------------------------------------------------------------

Enter the phrase query: dsfajkadsjkldf
No matching documents found.

-------------------------------------------------------------------------------------------

Analysis:
All the outputs seem as expected and if you inspect the documents that are returned then
you will find the words from the phrase query present in the documents.
In testing I used a dataset with only 6 documents and if the query was one word that was only
present in one doc then it would be the only document with a score of above zero.

The cosine similarity is very similar to the tfidf score and with the same weighting scheme they are even more similar
however some do have a different 5th document returned.
for example, using binary weighting scheme, the tfidf returns:
life.txt, mydream.txt, stairdre.txt, emperor3.txt, plescopm.txt

and in comparrison the binary cosine similarity returns:
life.txt, mydream.txt, stairdre.txt, emperor3.txt, discocanbefun.txt

while the scores do differ a tiny bit, the only ranking that changes is the 5th document.
