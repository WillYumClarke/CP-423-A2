"""
CP 423 Text Retrieval and Search Engines
Assignment 2
William Clarke 190524800
Andrew Best 190620060
"""
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from math import log
from sklearn.metrics.pairwise import cosine_similarity


def main():
    # Load and preprocess the dataset
    data_folder = 'data'
    dataset = []
    doc_ids = {}  # Dictionary to store document IDs

    # reat the contents of all the files inside the Data folder and add them to the dataset (contains duplicated for TF)
    for file_id, file_name in enumerate(os.listdir(data_folder), start=1):
        file_path = os.path.join(data_folder, file_name)
        try:
            with open(file_path, 'r') as file:
                # print(file_path)
                text = file.read()
                # use preprocess_text function to process the tokens
                tokens = preprocess_text(text)
                dataset.append(tokens)
                doc_ids[file_id] = file_name
                file.close()
        except:
            file.close()

    # Development of positional index data structure
    positional_index = {}

    for doc_id, tokens in enumerate(dataset):
        for position, token in enumerate(tokens):
            if token not in positional_index:
                positional_index[token] = {}

            if doc_id not in positional_index[token]:
                positional_index[token][doc_id] = []

            positional_index[token][doc_id].append(position)
    vocabulary = sorted(positional_index.keys())

    # Calculate the TF-IDF matrix for each weighting scheme
    print("Calculating TF-IDF matrices...")
    tfidf_matrix_binary = calculate_tfidf_matrix(
        dataset, positional_index, vocabulary, "Binary")
    print("Binary weighting scheme matrix complete")
    tfidf_matrix_raw_count = calculate_tfidf_matrix(
        dataset, positional_index, vocabulary, "Raw Count")
    print("Raw Count weighting scheme matrix complete")
    tfidf_matrix_term_freq = calculate_tfidf_matrix(
        dataset, positional_index, vocabulary, "Term Frequency")
    print("Term Frequency weighting scheme matrix complete")
    tfidf_matrix_log_norm = calculate_tfidf_matrix(
        dataset, positional_index, vocabulary, "Log Normalization")
    print("Log Normalization weighting scheme matrix complete")
    # tfidf_matrix_double_norm = calculate_tfidf_matrix(dataset, positional_index, vocabulary, "Double Normalization")
    print("Double Normalization weighting scheme matrix complete")
    print("Matrices calculated")

    # User entered Query section, loops for multiple querys
    while True:
        query = input("Enter the phrase query: ")
        query_tokens = preprocess_text(query)  # process the query into tokens

        if len(query_tokens) <= 5:  # ensure query is proper length
            top_indices_binary, top_scores_binary, returned_docs_binary = search_phrase_tfidf(
                positional_index, query_tokens, tfidf_matrix_binary, vocabulary, dataset, doc_ids, "Raw Count")
            top_indices_raw_count, top_scores_raw_count, returned_docs_raw_count = search_phrase_tfidf(
                positional_index, query_tokens, tfidf_matrix_raw_count, vocabulary, dataset, doc_ids, "Binary")
            top_indices_term_freq, top_scores_term_freq, returned_docs_term_freq = search_phrase_tfidf(
                positional_index, query_tokens, tfidf_matrix_term_freq, vocabulary, dataset, doc_ids, "Term Frequency")
            top_indices_log_norm, top_scores_log_norm, returned_docs_log_norm = search_phrase_tfidf(
                positional_index, query_tokens, tfidf_matrix_log_norm, vocabulary, dataset, doc_ids, "Log Normalization")
            # top_indices_double_norm, top_scores_double_norm, returned_docs_double_norm = search_phrase_tfidf(positional_index, query_tokens, tfidf_matrix_double_norm, vocabulary, dataset, doc_ids, "Double Normalization")
            if len(top_indices_binary) > 0:
                print(
                    "\nTop 5 relevant documents based on TF-IDF score with binary term frequency weighting scheme:")
                i = 0
                for doc_id, score in zip(top_indices_binary, top_scores_binary):
                    for key, value in doc_ids.items():
                        if value == doc_id:
                            print(
                                f"{top_indices_binary[i]} DocID: {key}, Score: {score}")
                            i += 1
                print(f"Total docs returned: {returned_docs_binary}")

                print(
                    "\nTop 5 relevant documents based on TF-IDF score with raw count term frequency weighting scheme:")
                i = 0
                for doc_id, score in zip(top_indices_raw_count, top_scores_raw_count):
                    for key, value in doc_ids.items():
                        if value == doc_id:
                            print(
                                f"{top_indices_raw_count[i]} Doc ID: {key}, Score: {score}")
                            i += 1
                print(f"Total docs returned: {returned_docs_raw_count}")

                print(
                    "\nTop 5 relevant documents based on TF-IDF score with term frequency term frequency weighting scheme:")
                i = 0
                for doc_id, score in zip(top_indices_term_freq, top_scores_term_freq):
                    for key, value in doc_ids.items():
                        if value == doc_id:
                            print(
                                f"{top_indices_term_freq[i]} Doc ID: {key}, Score: {score}")
                            i += 1
                print(f"Total docs returned: {returned_docs_term_freq}")

                print(
                    "\nTop 5 relevant documents based on TF-IDF score with log normalization term frequency weighting scheme:")
                i = 0
                for doc_id, score in zip(top_indices_log_norm, top_scores_log_norm):
                    for key, value in doc_ids.items():
                        if value == doc_id:
                            print(
                                f"{top_indices_log_norm[i]} Doc ID: {key}, Score: {score}")
                            i += 1
                print(f"Total docs returned: {returned_docs_log_norm}")

                """print(
                    "\nTop 5 relevant documents based on TF-IDF score with double normalization term frequency weighting scheme:")
                i=0
                for doc_id, score in zip(top_indices_double_norm, top_scores_double_norm):
                    for key, value in doc_ids.items():
                        if value == doc_id:
                            print(f"{top_indices_double_norm} Doc ID: {key}, Score: {score}")
                            i+=1
                print(f"Total docs returned: {returned_docs_double_norm}")"""
                
                top_indices_cosine_binary, top_scores_cosine_binary, returned_docs_cosine_binary = search_phrase_cosine_similarity(
                    query_tokens, tfidf_matrix_binary, vocabulary, doc_ids)
                top_indices_cosine_raw_count, top_scores_cosine_raw_count, returned_docs_cosine_raw_count = search_phrase_cosine_similarity(
                    query_tokens, tfidf_matrix_raw_count, vocabulary, doc_ids)
                top_indices_cosine_term_freq, top_scores_cosine_term_freq, returned_docs_cosine_term_freq = search_phrase_cosine_similarity(
                    query_tokens, tfidf_matrix_term_freq, vocabulary, doc_ids)
                top_indices_cosine_log_norm, top_scores_cosine_log_norm, returned_docs_cosine_log_norm = search_phrase_cosine_similarity(
                    query_tokens, tfidf_matrix_log_norm, vocabulary, doc_ids)
                """top_indices_cosine_double_norm, top_scores_cosine_double_norm, returned_docs_cosine_double_norm = search_phrase_cosine_similarity(
                    query_tokens, tfidf_matrix_double_norm, vocabulary, doc_ids)"""

                if len(top_indices_cosine_binary) > 0:
                    i = 0
                    print(
                        "\nTop 5 relevant documents based on cosine similarity and binary weighting scheme:")
                    for doc_id, score in zip(top_indices_cosine_binary, top_scores_cosine_binary):
                        # Print the cosine similarity results
                        for key, value in doc_ids.items():
                            if value == doc_id:
                                print(
                                    f"{top_indices_cosine_binary[i]} Doc ID: {key}, Score: {score}")
                                i += 1
                    print(
                        f"Total docs returned: {returned_docs_cosine_binary}")

                    print(
                        "\nTop 5 relevant documents based on cosine similarity and raw count weighting scheme:")
                    i = 0
                    for doc_id, score in zip(top_indices_cosine_raw_count, top_scores_cosine_raw_count):
                        # Print the cosine similarity results
                        for key, value in doc_ids.items():
                            if value == doc_id:
                                print(
                                    f"{top_indices_cosine_raw_count[i]}Doc ID: {key}, Score: {score}")
                                i += 1
                    print(
                        f"Total docs returned: {returned_docs_cosine_raw_count}")

                    print(
                        "\nTop 5 relevant documents based on cosine similarity and term frequency weighting scheme:")
                    i = 0
                    for doc_id, score in zip(top_indices_cosine_term_freq, top_scores_cosine_term_freq):
                        # Print the cosine similarity results
                        for key, value in doc_ids.items():
                            if value == doc_id:
                                print(
                                    f"{top_indices_cosine_term_freq[i]}Doc ID: {key}, Score: {score}")
                                i += 1
                    print(
                        f"Total docs returned: {returned_docs_cosine_term_freq}")

                    print(
                        "\nTop 5 relevant documents based on cosine similarity and log normalization weighting scheme:")
                    i = 0
                    for doc_id, score in zip(top_indices_cosine_log_norm, top_scores_cosine_log_norm):
                        # Print the cosine similarity results
                        for key, value in doc_ids.items():
                            if value == doc_id:
                                print(
                                    f"{top_indices_cosine_log_norm[i]} Doc ID: {key}, Score: {score}")
                                i += 1
                    print(
                        f"Total docs returned: {returned_docs_cosine_log_norm}")

                    """for doc_id, score in zip(top_indices_cosine_double_norm, top_scores_cosine_double_norm):
                        # Print the cosine similarity results
                        for key, value in doc_ids.items():
                            if value == doc_id:
                                print(f"Doc ID: {key}, Score: {score}")
                    print(f"Total docs returned: {returned_docs_cosine_double_norm}")"""

            else:
                print("No matching documents found.")
        else:
            print("Query length must be less than or equal to than 5.")
        print("\n")
    return


# Preprocessing tasks
def preprocess_text(text):
    # Transform text to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Remove punctuation marks and empty space tokens
    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]
    tokens = [token for token in tokens if token != '']

    return tokens


def calculate_tfidf_matrix(dataset, positional_index, vocabulary, weighting_scheme):
    num_docs = len(dataset)
    vocab_size = len(vocabulary)
    tfidf_matrix = np.zeros((num_docs, vocab_size))
    doc_lengths = np.zeros(num_docs)
    i = 0
    for doc_id, tokens in enumerate(dataset):
        for term_index, term in enumerate(vocabulary):
            term_frequency = calculate_term_frequency(
                tokens, term, weighting_scheme)
            document_frequency = len(positional_index.get(term, {}))
            idf = calculate_inverse_document_frequency(
                document_frequency, num_docs)
            tfidf_matrix[doc_id, term_index] = term_frequency * \
                idf  # Matrix construction
            doc_lengths[doc_id] += (term_frequency * idf) ** 2
            i += 1
    doc_lengths = np.sqrt(doc_lengths)
    # Add a small value to avoid division by zero
    tfidf_matrix = tfidf_matrix / (doc_lengths[:, np.newaxis] + 1e-7)

    return tfidf_matrix


# Calculate term frequency using different weighting schemes
def calculate_term_frequency(tokens, token, weighting_scheme):
    term_frequency = 0
    # Change the value inside calculate_term_frequency function to select desired weighting scheme
    if weighting_scheme == "Binary":
        if token in tokens:
            term_frequency = 1
    elif weighting_scheme == "Raw Count":
        term_frequency = tokens.count(token)
    elif weighting_scheme == "Term Frequency":
        term_frequency = tokens.count(token) / len(tokens)
    elif weighting_scheme == "Log Normalization":
        term_frequency = log(1 + tokens.count(token))
    elif weighting_scheme == "Double Normalization":
        for word in tokens:
            if word == token:
                term_frequency += 1
        max_frequency = max([tokens.count(t) for t in set(tokens)])
        term_frequency = 0.5 + \
            (0.5 * (term_frequency / max_frequency)) if term_frequency > 0 else 0

    return term_frequency


def calculate_inverse_document_frequency(document_frequency, num_docs):
    return log(num_docs / (document_frequency + 1))


# Search phrase queries using TF-IDF scores
def search_phrase_tfidf(positional_index, query_tokens, tfidf_matrix, vocabulary, dataset, doc_ids, weighting_scheme, num_results=5):
    query_vector = np.zeros(len(vocabulary))
    for term_index, term in enumerate(vocabulary):
        term_frequency = calculate_term_frequency(
            query_tokens, term, weighting_scheme)
        document_frequency = len(positional_index.get(term, {}))
        idf = calculate_inverse_document_frequency(
            document_frequency, len(dataset))
        query_vector[term_index] = term_frequency * idf

    if np.count_nonzero(query_vector) == 0:  # case for no docs relevant
        return [], [], 0

    # Normalize the query vector
    query_vector = query_vector / np.linalg.norm(query_vector, ord=2)
    scores = np.dot(tfidf_matrix, query_vector)
    top_indices = np.argsort(scores)[::-1][:num_results]
    top_scores = scores[top_indices]

    returned_docs = np.count_nonzero(scores > 0)

    return [doc_ids[idx+1] for idx in top_indices], top_scores, returned_docs


def search_phrase_cosine_similarity(query_tokens, tfidf_matrix, vocabulary, doc_ids, num_results=5):
    query_vector = np.zeros(tfidf_matrix.shape[1])
    for term in query_tokens:
        if term in vocabulary:
            term_index = vocabulary.index(term)
            query_vector[term_index] = 1

    if np.count_nonzero(query_vector) == 0:  # Case for if no docs are relevant
        return [], [], 0

    # Normalize the query vector
    query_vector = query_vector / np.linalg.norm(query_vector, ord=2)
    similarity_scores = cosine_similarity(tfidf_matrix, [query_vector])
    top_indices = np.argsort(similarity_scores.flatten())[::-1][:num_results]
    top_scores = similarity_scores[top_indices]
    
    returned_docs = np.count_nonzero(similarity_scores > 0)

    return [doc_ids[idx+1] for idx in top_indices], top_scores.flatten(), returned_docs



main()
