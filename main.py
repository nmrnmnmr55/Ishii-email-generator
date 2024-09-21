import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import anthropic
from bs4 import BeautifulSoup
import os
import re
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

CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# APIキーの確認
if not CLAUDE_API_KEY:
    st.error("Claude API key is not set. Please check your .env file.")
    st.stop()

# Claude クライアントの初期化
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

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

def generate_claude_response(prompt, system_message="You are a helpful assistant."):
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            system=system_message,
            messages=[
                {"role": "user", "content": prompt + "\n\nPlease provide a complete and coherent response without any greeting or closing remarks. Start your response with the main content. For better readability, please add appropriate line breaks after each complete thought or every 2-3 sentences."}
            ]
        )
        return message.content[0].text
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
    # 余分な空白や改行を削除
    response = re.sub(r'\s+', ' ', response).strip()
    
    # 最初の文章が完全でない場合、削除する
    first_sentence_end = response.find('。')
    if first_sentence_end != -1:
        first_sentence = response[:first_sentence_end+1]
        if len(first_sentence.split()) < 3:  # 短すぎる文は削除
            response = response[first_sentence_end+1:].strip()
    
    # 適切な位置に改行を挿入
    sentences = re.split('(。|！|？)', response)
    formatted_content = ''
    sentence_count = 0
    for i in range(0, len(sentences) - 1, 2):
        formatted_content += sentences[i] + sentences[i+1]
        sentence_count += 1
        if sentence_count % 2 == 0:
            formatted_content += '\n\n'
    
    formatted_response = f"{customer_name}様\n\n"
    formatted_response += "平素より大変お世話になっております。\n\n"
    formatted_response += formatted_content.strip() + "\n\n"
    formatted_response += "引き続き何卒よろしくお願い申し上げます。\n\n"
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

        if matching_keywords:
            responses = []
            st.text("Matching keywords found in the email: " + ", ".join([keyword for keyword, _ in matching_keywords]))
            for keyword, url in matching_keywords:
                url_content = get_url_content(url)
                response = generate_claude_response(
                    f"Based on this information about {keyword}: {url_content}, write a response to: {patient_email}",
                    "You are Dr. Ishii, a professional and caring beauty clinic doctor. Respond to the patient's email based on the given information. Provide a complete and coherent response."
                )
                if response:
                    responses.append(response)
            
            combined_response = " ".join(responses)
        else:
            combined_response = generate_claude_response(
                f"Write a response to this patient email: {patient_email}",
                "You are Dr. Ishii, a professional and caring beauty clinic doctor. Respond to the patient's email in a helpful and informative manner. Provide a complete and coherent response."
            )

        if combined_response:
            similar_email_index = find_similar_email(combined_response, vectorizer, tfidf_matrix)
            if similar_email_index is not None and past_emails_list:
                similar_email = past_emails_list[similar_email_index]
                final_response = generate_claude_response(
                    f"Rewrite this response in Dr. Ishii's tone and style, based on this example: {similar_email}\n\nResponse to rewrite: {combined_response}",
                    "You are an AI assistant helping Dr. Ishii maintain consistency in email communications. Rewrite the given response to match Dr. Ishii's tone and style. Provide a complete and coherent response."
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