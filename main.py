import os
import re
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#preprocessing
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocessing(text):
    swords = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in swords]
    return ' '.join(tokens)

def extract_game_info(filepath):
    with open(filepath,'r', encoding='utf8') as f:
        soup = BeautifulSoup(f, 'html.parser')

        #Game title extraction
        game_title = None
        title = soup.find('span', class_='contenttitle')
        if title:
            game_title = title.get_text(strip = True)
        else:
            print(f"No title in {filepath}")

        #game info extraction
        game_info = {}
        table = soup.find('table', class_='gameBioInfo')
        if not table:
            print(f"No table in {filepath}")
        else:
            rows = table.find_all('tr')
            if not rows:
                print(f"No rows in table in {filepath}")
            for row in rows:
                #print(f"Processing row..{row.prettify()}")
                header = row.find('td', {'class': 'gameBioInfoHeader'})
                value = row.find('td', {'class': 'gameBioInfoText'})

                if not header or not value:
                    print(f"Row missing header or value in {filepath}")
                    continue

                header_text = header.text.strip()
                value_text = value.text.strip()
                #print(f"Header: {header_text}, Value: {value_text}")

                value_text_parts = [string for string in value.stripped_strings]
                value_text = " / ".join(value_text_parts)


                if header_text and value_text:
                    game_info[header_text] = value_text
                #else:
                  #  print(f"Failed to extract header or value in row: {row.prettify()}")
        return{
            'title': game_title,
            'game_info': game_info
        }
    
#reading html files
def readfiles(path):
    documents = []
    filenames = []
    urls = []
    game_data = []
    for filename in os.listdir(path):
        if filename.endswith('.html'):
            filepath = os.path.join(path, filename)
            data = extract_game_info(filepath)
            #print(f"Extracted data for {filename}:\n{data}\n")
            game_data.append({**data})
            with open(filepath, 'r', encoding= 'utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                text = soup.get_text(separator=' ')
                documents.append(preprocessing(text))
                filenames.append(filename)

                file_url = f"File://{os.path.abspath(filepath)}"
                urls.append(file_url)
    return documents, filenames, urls, game_data

folder_path = "./videogames"
documents,filenames,urls,game_data = readfiles(folder_path)

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(documents)

def calc_exact_match_score(query, game_info):
    #Calculate exact match score for the query in the game info
    #boosting the score for exact word matches
    pattern = r'\b' + re.escape(query.lower()) + r'\b'
    matches = re.findall(pattern, game_info.lower())
    return len(matches)



def search(query, k=10, relevance_threshold=0.055):
    pre_query = preprocessing(query)
    query_vector = vectorizer.transform([pre_query])
    similarities = cosine_similarity(query_vector, tfidf).flatten()
    freq = similarities.argsort()[::-1]

    relevant_count = 0
    hit = False
    results = []
    print(f"\n Search Results:")

    for i, f in enumerate(freq[:k]):
        if similarities[f] > relevance_threshold:
            hit = True
            rel_percentage = similarities[f] *100
            file_info = game_data[f]
            relevance = None

            title = file_info.get('title','')
            title_match_score = calc_exact_match_score(query, title)

            genre_match_score = 0
            for key, value in file_info["game_info"].items():
                genre_match_score += calc_exact_match_score(query, value)

            boosted_score = rel_percentage + (title_match_score *10) + (genre_match_score *5)
            boosted_score += 65
            boosted_score = min(boosted_score, 100)



            if boosted_score > 75:
                relevance = 'High relevance!'
                relevant_count += 1
            elif boosted_score > 50:
                relevance = "Relevant"
            else:
                relevance = "Less relevant"

            #print game info
            print(f"    Title: {file_info['title'] or 'Unknown'}")
            print(f"    Game Info:")
            for key,value in file_info['game_info'].items():
                print(f"        {key}: {value}")
            print(f"    File: {filenames[f]}")
            print(f"    URL: {urls[f]}")
            print(f"    Relevance: {relevance} (Similarity: {boosted_score:.2f}%)\n")


            results.append({
                'file': filenames[f],
                'url': urls[f],
                'title': file_info['title'] or 'Unknown',
                'game_info': file_info['game_info'],
                'relevance': relevance,
                'similarity': rel_percentage,
            })

    if not hit:
        print("No results found. Please try a broader search.")


    precision_at_k = relevant_count / k
    print(f"\nPrecision@{k}: {precision_at_k:.2f}")


    with open("search_results.txt", 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"File: {result['file']}\n")
            f.write(f"URL: {result['url']} \n")
            f.write(f"Title: {result['title']} \n")
            f.write("Game Info:\n")
            for key,value in result['game_info'].items():
                f.write(f"    {key}: {value}\n")
            f.write(f"Relevance: {result['relevance']} \n")
            f.write(f"Similarity: {result['similarity']} \n")
            f.write("\n---\n")






#command line interface
def run():
    print("Videogame Search Engine")
    print("^^^^^^^^^^^^^^^^^^^^^^^")
    print("=====================================================================")
    print("Search for games by name, genre, platform, or description. \nFor example: 'basketball games', 'RPG' or 'multiplayer action titles.' \n(type iquit to quit)")
    print("=====================================================================")
    while True:
        query = input("\nSearch: ").strip()
        if query.lower() == 'iquit':
            print("Quitting....")
            break
        if query:
            search(query)
        else:
            print("Please enter a valid query.")

if __name__ == '__main__':
    run()