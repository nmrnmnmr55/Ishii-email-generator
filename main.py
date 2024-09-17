import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from openai import OpenAI
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
import os
import re
import tiktoken
from dotenv import load_dotenv

# キーワードとURLのデータベース
url_database = {
    'エクソソーム点滴': 'https://smartskin-clinic.com/menu/injection-infusion/exosome-infusion/',
    'エクソソーム点鼻': 'https://smartskin-clinic.com/menu/exosome-nasalspray/',
    'NMN点滴': 'https://smartskin-clinic.com/menu/injection-infusion/nmn/',
    'ジュベルック and 水光': 'https://smartskin-clinic.com/menu/suikou/juvelook/',
    'エクソソーム and 水光': 'https://smartskin-clinic.com/menu/suikou/exosome/',
    'キアラレジュ and 水光': 'https://smartskin-clinic.com/menu/suikou/kiarareju/',
    'リズネ': 'https://smartskin-clinic.com/menu/injection-infusion/lizne/',
    'ネオファウンド': 'https://smartskin-clinic.com/menu/suikou/neofound/',
    'スキンボト': 'https://smartskin-clinic.com/menu/suikou/skinbotox/',
    'スネコス': 'https://smartskin-clinic.com/menu/injection-infusion/sunekos/',
    'スネコスパフォルマ': 'https://smartskin-clinic.com/menu/injection-infusion/sunekos-performa/',
    'スネコスセル': 'https://smartskin-clinic.com/menu/sunekos-cell/',
    'プロファイロ': 'https://smartskin-clinic.com/menu/injection-infusion/profhilo/',
    'ジュベルック': 'https://smartskin-clinic.com/menu/injection-infusion/juvelook/',
    'ジュベルックボリューム or レニスナ': 'https://smartskin-clinic.com/menu/juvelookvolume-lenisna/',
    'ゴウリ or GOURI': 'https://smartskin-clinic.com/menu/gouri/',
    'ジャルプロ or JALUPRO': 'https://smartskin-clinic.com/menu/injection-infusion/jalupro/',
    'リッチPL': 'https://smartskin-clinic.com/menu/injection-infusion/richpltopic/',
    'リジュラン or サーモン': 'https://smartskin-clinic.com/menu/injection-infusion/rejuran/',
    'ショッピングリフト or ショッピングスレッド': 'https://smartskin-clinic.com/menu/shoppinglift/',
    'アイスレッド': 'https://smartskin-clinic.com/menu/eyethread/',
    'イソトレ': 'https://smartskin-clinic.com/isotretinoin/',
    'ダーマペン': 'https://smartskin-clinic.com/dermapen-exosome/',
    'ヴェルベット': 'https://smartskin-clinic.com/menu/velvetskin/',
    'フォト': 'https://smartskin-clinic.com/photofacial/'
}

# 環境変数の設定
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# APIキーの確認
if not OPENAI_API_KEY:
    st.error("OpenAI API key is not set. Please check your .env file.")
    st.stop()

# OpenAI クライアントの初期化
client = OpenAI(api_key=OPENAI_API_KEY)

# トークンカウンターの初期化
enc = tiktoken.encoding_for_model("gpt-4o-mini")

def count_tokens(text):
    return len(enc.encode(text))

def calculate_price(input_tokens, output_tokens):
    input_price = (input_tokens / 1000000) * 0.15
    output_price = (output_tokens / 1000000) * 0.60
    return input_price + output_price

def load_past_emails(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().split('\n\n')  # 各メールは空行で区切られていると仮定
    except FileNotFoundError:
        st.error(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return []

def create_tfidf_vectors(documents):
    if documents:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        return vectorizer, tfidf_matrix
    return None, None

def find_similar_email(new_email, vectorizer, tfidf_matrix):
    if vectorizer is not None and tfidf_matrix is not None and tfidf_matrix.shape[0] > 0:
        new_email_vector = vectorizer.transform([new_email])
        similarities = cosine_similarity(new_email_vector, tfidf_matrix)
        most_similar_index = similarities.argmax()
        return most_similar_index
    return None

def get_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error fetching URL content: {e}")
        return ""

def google_search(query):
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    res = service.cse().list(q=query, cx=GOOGLE_CSE_ID).execute()
    return res['items'] if 'items' in res else []

def scrape_search_results(query):
    search_results = google_search(query)
    scraped_content = []
    for item in search_results[:3]:  # 最初の3つの結果のみスクレイピング
        url = item['link']
        content = get_url_content(url)
        scraped_content.append(content)
    return " ".join(scraped_content)

def generate_gpt4o_response(prompt, system_message="You are a helpful assistant."):
    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt + "\n\nPlease do not include any greeting, signature, or closing remarks at the beginning or end of your response."}
        ]
        input_tokens = sum(count_tokens(msg["content"]) for msg in messages)
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # または利用可能な正しいモデル名
            messages=messages
        )
        
        output_tokens = count_tokens(completion.choices[0].message.content)
        price = calculate_price(input_tokens, output_tokens)
        
        st.write(f"API Call Stats:")
        st.write(f"Input tokens: {input_tokens}")
        st.write(f"Output tokens: {output_tokens}")
        st.write(f"Estimated price: ${price:.6f}")
        
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error in API call: {str(e)}")
        return None

def find_matching_keywords(text):
    matches = []
    for key, url in url_database.items():
        if ' and ' in key.lower():
            keywords = key.lower().split(' and ')
            if all(keyword.strip() in text.lower() for keyword in keywords):
                matches.append((key, url))
        elif ' or ' in key.lower():
            keywords = key.lower().split(' or ')
            if any(keyword.strip() in text.lower() for keyword in keywords):
                matches.append((key, url))
        else:
            if key.lower() in text.lower():
                matches.append((key, url))
    return matches

def format_response(response, customer_name):
    # 挨拶、署名、締めくくりの言葉を削除
    response = re.sub(r'^.*様\s*', '', response, flags=re.DOTALL)  # 冒頭の挨拶を削除
    response = re.sub(r'\n敬具.*', '', response, flags=re.DOTALL)
    response = re.sub(r'\nSmart skin CLINIC.*', '', response, flags=re.DOTALL)
    
    formatted_response = f"{customer_name}様\n"
    formatted_response += "平素より大変お世話になっております。\n\n"
    formatted_response += response.strip() + "\n\n"
    formatted_response += "引き続き何卒よろしくお願い申し上げます。\n"
    formatted_response += "Smart skin CLINIC\n"
    formatted_response += "院長　石井"
    return formatted_response

def extract_customer_name(email_content):
    # 簡単な例: 最初の行に「〇〇様」という形式で名前があると仮定
    first_line = email_content.split('\n')[0]
    match = re.search(r'(.+)様', first_line)
    if match:
        return match.group(1)
    return "お客様"  # デフォルトの名前

def main():
    st.title("Dr. Ishii's Email Response Generator")

    # 石井先生の過去のメールをファイルから読み込む
    past_emails_list = load_past_emails('ishii_past_emails.txt')
    vectorizer, tfidf_matrix = create_tfidf_vectors(past_emails_list)

    patient_email = st.text_area("Enter patient's email content:")

    if st.button("Generate Response"):
        customer_name = extract_customer_name(patient_email)
        matching_keywords = find_matching_keywords(patient_email)

        total_input_tokens = 0
        total_output_tokens = 0
        total_price = 0

        if matching_keywords:
            responses = []
            st.text("Matching keywords found in the email: " + ", ".join([keyword for keyword, _ in matching_keywords]))
            for keyword, url in matching_keywords:
                url_content = get_url_content(url)
                scraped_content = scrape_search_results(keyword)
                combined_info = f"{url_content}\n\nAdditional information from web search:\n{scraped_content}"
                response = generate_gpt4o_response(
                    f"Based on this information about {keyword}: {combined_info}, write a response to: {patient_email}",
                    "You are Dr. Ishii, a professional and caring beauty clinic doctor. Respond to the patient's email based on the given information."
                )
                if response:
                    responses.append(response)
            
            combined_response = " ".join(responses)
        else:
            combined_response = generate_gpt4o_response(
                f"Write a response to this patient email: {patient_email}",
                "You are Dr. Ishii, a professional and caring beauty clinic doctor. Respond to the patient's email in a helpful and informative manner."
            )

        if combined_response:
            similar_email_index = find_similar_email(combined_response, vectorizer, tfidf_matrix)
            if similar_email_index is not None and past_emails_list:
                similar_email = past_emails_list[similar_email_index]
                final_response = generate_gpt4o_response(
                    f"Rewrite this response in Dr. Ishii's tone and style, based on this example: {similar_email}\n\nResponse to rewrite: {combined_response}",
                    "You are an AI assistant helping Dr. Ishii maintain consistency in email communications. Rewrite the given response to match Dr. Ishii's tone and style."
                )
            else:
                final_response = combined_response

            if final_response:
                formatted_response = format_response(final_response, customer_name)
                st.subheader("Generated Response:")
                st.text(formatted_response)
            else:
                st.error("Failed to generate a response. Please try again.")
        else:
            st.error("Failed to generate initial response. Please try again.")

if __name__ == "__main__":
    main()