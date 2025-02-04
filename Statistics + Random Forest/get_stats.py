import os
import json
import pandas as pd
import re
import spacy

# De aici: https://en.wiktionary.org/wiki/Category:Romanian_prefixes
romanian_prefixes = [
    # A
    "agro", "alt", "ante", "anti", "aorto", "arhi", "astro",

    # B
    "balano",

    # C
    "cardio", "carpo", "cosmo",

    # D
    "demono", "des", "dez",

    # F
    "franco",

    # G
    "gastro", "germano", "greco",

    # H
    "hecto", "hiper",

    # I
    "în",

    # K
    "kilo",

    # L
    "lexico",

    # M
    "mili", "muzico",

    # N
    "nano", "ne",

    # O
    "ori", "ornito",

    # P
    "pneumo", "pre", "prea", "proto", "pseudo", "psiho",

    # R
    "răs", "re", "rino", "ruso",

    # S
    "stră", "sub",

    # T
    "tehno", "teo", "termo",

    # V
    "vice"
]

def replace_i_prefix(word, prefixes):
  for prefix in prefixes:
    try:
      if word.lower().startswith(prefix) and len(word) > len(prefix) and word[len(prefix):][0] in ["î", "Î"]:
        first_letter = word[len(prefix):][0]
        first_letter = "i" if first_letter == "î" else ("I" if first_letter == "Î" else first_letter)
        word = prefix + first_letter + word[len(prefix) + 1:]

    except:
      print(word)

  word = word.replace("î", "a").replace("Î", "A")

  return word

def no_diacritics(text, prefixes):

  text = replace_i_prefix(text, prefixes)


  text = text.replace("â", "i")
  text = text.replace("Â", "I")
  text = text.replace("ș", "s")
  text = text.replace("ş", "s")
  text = text.replace("Ș", "S")
  text = text.replace("Ş", "S")
  text = text.replace("ț", "t")
  text = text.replace("ţ", "t")
  text = text.replace("Ț", "T")
  text = text.replace("Ţ", "T")

  # If î is the first letter of the word, replace it with i
  if text.startswith("î"):
    text = text.replace("î", "i")
  if text.startswith("Î"):
    text = text.replace("Î", "I")
  # If the last letter of the word is î, replace it with i
  if text.endswith("î"):
    text = text.replace("î", "i")
  if text.endswith("Î"):
    text = text.replace("Î", "I")
  # Else replace î with a
  if "î" in text:
    text = text.replace("î", "a")
  # text = text.replace("î", "i")
  # text = text.replace("Î", "I")
  text = text.replace("ă", "a")
  text = text.replace("Ă", "A")

  return text

romanian=[
    "a", "abia", "acea", "aceasta", "această", "aceea", "aceeasi", "acei",
    "aceia", "acel", "acela", "acelasi", "acele", "acelea", "acest", "acesta",
    "aceste", "acestea", "acestei", "acestia", "acestui", "aceşti", "aceştia",
    "acești", "aceștia", "acolo", "acord", "acum", "adica", "ai", "aia",
    "aibă", "aici", "aiurea", "al", "ala", "alaturi", "ale", "alea", "alt",
    "alta", "altceva", "altcineva", "alte", "altfel", "alti", "altii", "altul",
    "alături", "am", "anume", "apoi", "ar", "are", "as", "asa", "asemenea",
    "asta", "astazi", "astea", "astfel", "astăzi", "asupra", "atare", "atat",
    "atata", "atatea", "atatia", "ati", "atit", "atita", "atitea", "atitia",
    "atunci", "au", "avea", "avem", "aveţi", "aveți", "avut", "azi", "aş",
    "aşadar", "aţi", "aș", "așadar", "ați", "b", "ba", "bine", "bucur", "bună",
    "c", "ca", "cam", "cand", "capat", "care", "careia", "carora", "caruia",
    "cat", "catre", "caut", "ce", "cea", "ceea", "cei", "ceilalti", "cel",
    "cele", "celor", "ceva", "chiar", "ci", "cinci", "cind", "cine", "cineva",
    "cit", "cita", "cite", "citeva", "citi", "câțiva", "conform", "contra",
    "cu", "cui", "cum", "cumva", "curând", "curînd", "când", "cât", "câte",
    "câtva", "câţi", "câți", "cînd", "cît", "cîte", "cîtva", "cîţi", "cîți",
    "că", "căci", "cărei", "căror", "cărui", "către", "d", "da", "daca",
    "dacă", "dar", "dat", "datorită", "dată", "dau", "de", "deasupra", "deci",
    "decit", "degraba", "deja", "deoarece", "departe", "desi", "despre",
    "deşi", "deși", "din", "dinaintea", "dintr", "dintr-", "dintre", "doar",
    "doi", "doilea", "două", "drept", "dupa", "după", "dă", "e", "ea", "ei",
    "el", "ele", "era", "eram", "este", "eu", "exact", "eşti", "ești", "f",
    "face", "fara", "fata", "fel", "fi", "fie", "fiecare", "fii", "fim", "fiu",
    "fiţi", "fiți", "foarte", "fost", "frumos", "fără", "g", "geaba", "graţie",
    "grație", "h", "halbă", "i", "ia", "iar", "ieri", "ii", "il", "imi", "in",
    "inainte", "inapoi", "inca", "incit", "insa", "intr", "intre", "isi",
    "iti", "j", "k", "l", "la", "le", "li", "lor", "lui", "lângă", "lîngă",
    "m", "ma", "mai", "mare", "mea", "mei", "mele", "mereu", "meu", "mi",
    "mie", "mine", "mod", "mult", "multa", "multe", "multi", "multă", "mulţi",
    "mulţumesc", "mulți", "mulțumesc", "mâine", "mîine", "mă", "n", "ne",
    "nevoie", "ni", "nici", "niciodata", "nicăieri", "nimeni", "nimeri",
    "nimic", "niste", "nişte", "niște", "noastre", "noastră", "noi", "noroc",
    "nostri", "nostru", "nou", "noua", "nouă", "noştri", "noștri", "nu",
    "numai", "o", "opt", "or", "ori", "oricare", "orice", "oricine", "oricum",
    "oricând", "oricât", "oricînd", "oricît", "oriunde", "p", "pai", "parca",
    "patra", "patru", "patrulea", "pe", "pentru", "peste", "pic", "pina",
    "plus", "poate", "pot", "prea", "prima", "primul", "prin", "printr-",
    "putini", "puţin", "puţina", "puţină", "puțin", "puțina", "puțină", "până",
    "pînă", "r", "rog", "s", "sa", "sa-mi", "sa-ti", "sai", "sale", "sau",
    "se", "si", "sint", "sintem", "spate", "spre", "sub", "sunt", "suntem",
    "sunteţi", "sunteți", "sus", "sută", "sînt", "sîntem", "sînteţi",
    "sînteți", "să", "săi", "său", "t", "ta", "tale", "te", "ti", "timp",
    "tine", "toata", "toate", "toată", "tocmai", "tot", "toti", "totul",
    "totusi", "totuşi", "totuși", "toţi", "toți", "trei", "treia", "treilea",
    "tu", "tuturor", "tăi", "tău", "u", "ul", "ului", "un", "una", "unde",
    "undeva", "unei", "uneia", "unele", "uneori", "unii", "unor", "unora",
    "unu", "unui", "unuia", "unul", "v", "va", "vi", "voastre", "voastră",
    "voi", "vom", "vor", "vostru", "vouă", "voştri", "voștri", "vreme", "vreo",
    "vreun", "vă", "x", "z", "zece", "zero", "zi", "zice", "îi", "îl", "îmi",
    "împotriva", "în", "înainte", "înaintea", "încotro", "încât", "încît",
    "între", "întrucât", "întrucît", "îţi", "îți", "ăla", "ălea", "ăsta",
    "ăstea", "ăştia", "ăștia", "şapte", "şase", "şi", "ştiu", "ţi", "ţie",
    "șapte", "șase", "și", "știu", "ți", "ție"
]

# # Get all the words from the stop words list and apply the same transformation
stop_words = romanian
for i in range(len(stop_words)):
    stop_words[i] = no_diacritics(stop_words[i], romanian_prefixes)

stop_words = list(set(stop_words))

print(no_diacritics("cîțiva", romanian_prefixes))

# Load spaCy model
nlp = spacy.load("ro_core_news_sm")


def preprocess_text(text):
    """Preprocess the text before computing statistics."""
    text = no_diacritics(text, romanian_prefixes)
    text = ' '.join(word for word in text.split() if word.lower() not in stop_words)
    doc = nlp(text)
    text = ' '.join(token.lemma_ for token in doc if not token.is_stop)
    return text


def compute_text_metrics(text):
    """Extract comprehensive text metrics."""
    text = preprocess_text(text)  # Apply preprocessing

    # Tokenization
    words = re.findall(r'\w+', text.lower())
    sentences = re.split(r'[.!?]+', text)

    # Metrics computation
    return {
        'vocab_size': len(set(words)),
        'word_count': len(words),
        'type_token_ratio': len(set(words)) / len(words) if words else 0,
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
        'sentence_count': len(sentences),
        'words_per_sentence': len(words) / len(sentences) if sentences else 0
    }


def process_dataset(file_path):
    try:
        df = pd.read_json(file_path)
        df = df[df['text'].str.strip() != '']
    except:
        df = pd.read_json(file_path)
        df = df[df['content'].str.strip() != '']
        df.rename(columns={'content': 'text'}, inplace=True)

    # Apply metrics extraction
    metrics_df = df['text'].apply(compute_text_metrics).apply(pd.Series)

    # Combine with original dataframe
    result_df = pd.concat([df, metrics_df], axis=1)
    return result_df


f = open("stats.txt", "w")

for json_file in ['Banat.json', 'Spania.json', 'Ardeal.json', 'Germania.json', 'Bucovina.json',
                  'UK.json', 'Dobrogea.json', 'Crisana.json', 'Italia.json', 'Muntenia.json',
                  'Serbia.json', 'Maramures.json', 'Canada_EN.json', 'Moldova.json', 'Oltenia.json',
                  'Ucraina.json', 'Canada_Quebec.json', 'Sangerei.json', 'Hincesti.json', 'Causeni.json', 'Orhei.json', 'Criuleni.json', 'Balti.json', 'Ungheni.json', 'Ialoveni.json', 'Comrat.json', 'Calarasi.json', 'Cahul.json', 'Soroca.json']:
    file_path = "Dataset/" + json_file
    processed_data = process_dataset(file_path)

    # Display summary statistics
    print(processed_data[['vocab_size', 'word_count', 'type_token_ratio',
                          'avg_word_length', 'sentence_count', 'words_per_sentence']].describe())

    f.write("=" * 50 + "\n")
    f.write(json_file + "\n")
    f.write(str(processed_data[['vocab_size', 'word_count', 'type_token_ratio',
                                'avg_word_length', 'sentence_count',
                                'words_per_sentence']].describe()))
    f.write("\n")

f.close()
