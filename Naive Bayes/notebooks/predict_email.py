import pickle
import re
from nltk.stem.porter import PorterStemmer

# Load models from current directory
cv = pickle.load(open('cv.pkl', 'rb'))
clf = pickle.load(open('clf.pkl', 'rb'))

ps = PorterStemmer()

def clean_and_stem(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    stemmed = [ps.stem(word) for word in words]
    return " ".join(stemmed)

def predict_email(email_text):
    cleaned = clean_and_stem(email_text)
    vector = cv.transform([cleaned]).toarray()
    prediction = clf.predict(vector)[0]
    probability = clf.predict_proba(vector)[0]
    
    if prediction == 1:
        result = "SPAM"
        confidence = probability[0]
    else:
        result = "HAM"
        confidence = probability[1]
    
    print(f"\n{'='*60}")
    print(f"Email: {email_text}")
    print(f"{'='*60}")
    print(f"Result: {result}")
    print(f"Confidence: {confidence:.2%}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    while True:
        print("Enter 'quit' to exit")
        email = input("Enter email text: ").strip()
        
        if email.lower() == 'quit':
            print("Goodbye!")
            break
        
        if email:
            predict_email(email)
        else:
            print("Please enter some text!\n")