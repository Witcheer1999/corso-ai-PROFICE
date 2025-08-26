import string

def conta_parole(testo):
    parole = testo.split()
    return len(parole)

def conta_righe(testo):
    righe = testo.split('\n')
    return len(righe)

def parole_frequenti(testo):
    parole = testo.split()
    freq = {}
    for parola in parole:
        freq[parola] = freq.get(parola, 0) + 1
    frequenze = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return frequenze[:5]


if __name__ == "__main__":

    file_path = "C:\\desktopnoonedrive\\Ai Academy\\deposito-addari-EY\\18_08\\esercizio1\\misc_files\\testo_18_08.txt"

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            content = content.lower()
            content = content.translate(str.maketrans("", "", string.punctuation))
    except FileNotFoundError:
        print("File non trovato.")

    print("Numero di parole:", conta_parole(content))
    print("Numero di righe:", conta_righe(content))
    print("Parole frequenti:", parole_frequenti(content))