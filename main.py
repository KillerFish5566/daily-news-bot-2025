import os
import sys
import logging
import datetime
import warnings
from dotenv import load_dotenv

# --- 0. 設定與警告過濾 ---
warnings.filterwarnings("ignore", category=UserWarning, module="linebot")
# ddgs 的警告已經透過換套件解決了，所以這裡不需要再濾 duckduckgo

# 引入新版搜尋套件
from ddgs import DDGS

# 引入 Google GenAI SDK
from google import genai
from google.genai import types

# 引入 LINE BOT SDK v3
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    PushMessageRequest,
    TextMessage
)

# --- 1. 設定日誌系統 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- 2. 載入環境變數 ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID")

if not all([GEMINI_API_KEY, LINE_CHANNEL_ACCESS_TOKEN, LINE_USER_ID]):
    logger.error("❌ 環境變數缺失！請檢查 .env 檔案。")
    sys.exit(1)

# --- 3. 初始化 Google GenAI Client (關鍵修正！) ---
# 強制指定 api_version='v1'，避免去 v1beta 找不到 gemini-1.5-flash
client = genai.Client(api_key=GEMINI_API_KEY)

def get_target_date():
    return datetime.date.today()

def search_news(target_date, max_results_per_keyword=3):
    """
    搜尋國際新聞 (DeepResearch 邏輯)
    """
    date_str = target_date.strftime("%Y/%m/%d")
    logger.info(f"🔍 開始搜尋 {date_str} 的嚴肅國際新聞...")
    
    results = []
    
    # 搜尋關鍵字 (已排除娛樂內容)
    keywords = [
        "Major International Geopolitics -celebrity -gossip -sport -movie",
        "Global Economic Impact -stock -crypto",
        "Scientific Research Breakthroughs AI Space -movie -fiction"
    ]
    
    try:
        with DDGS() as ddgs:
            for query in keywords:
                logger.info(f"   正在搜尋分類: {query} ...")
                # timelimit='d' 代表過去一天
                news_gen = ddgs.news(query, region='wt-wt', safesearch='Off', timelimit='d')
                
                count = 0
                for r in news_gen:
                    if count >= max_results_per_keyword: break 
                    
                    title = r.get('title', '')
                    body = r.get('body', '')
                    url = r.get('url', '')
                    
                    # 二次過濾
                    block_list = ["Kardashian", "Taylor Swift", "Netflix", "Review", "Box Office"]
                    if any(bad_word in title for bad_word in block_list):
                        continue

                    if title and url:
                        results.append(f"類別: {query}\n標題: {title}\n摘要: {body}\n連結: {url}")
                        count += 1
                        
    except Exception as e:
        logger.error(f"⚠️ 搜尋時發生錯誤: {e}")
        return []

    logger.info(f"✅ 搜尋完成，共保留 {len(results)} 則高價值新聞。")
    return results

def generate_summary(news_list, target_date):
    """
    使用 Gemini 生成專業報告 (包含自動降級機制)
    """
    if not news_list:
        return None

    date_str = target_date.strftime("%Y/%m/%d")
    logger.info("🧠 Gemini 正在構思新聞報告...")

    prompt = (
        f"今天是 {date_str}。\n"
        "你現在是一位「專注於硬派國際局勢與前沿科學研究」的資深分析師。\n"
        "請根據以下搜集到的資料，整理出一份「高含金量」的日報。\n\n"
        "⛔ 嚴格過濾原則：\n"
        "1. 絕對不要包含娛樂、明星八卦、體育賽事、或是純粹的犯罪社會新聞。\n"
        "2. 如果資料中都是垃圾新聞，請直接回答「今日無重大地緣政治或科學新聞」。\n\n"
        "✅ 撰寫要求：\n"
        "1. 請挑選 5 則最具影響力的「地緣政治變動」或「重大科學發現」。\n"
        "2. 語氣要專業、客觀、精煉，像是在寫給 CEO 或研究員看的簡報。\n"
        "3. 格式：【領域標籤】標題 (換行) 深度摘要 (換行) 🔗 連結。\n"
        "4. 結尾請給一句關於「洞察世界」的專業短語。\n\n"
        "原始新聞資料：\n" + "\n---\n".join(news_list)
    )

    # 定義模型優先順序
    # 優先嘗試 Pro (品質最好)，失敗則退回 Flash (最穩)
    # 你可以把 'gemini-1.5-pro' 換成 'gemini-1.5-pro-002' 試試看，這通常是品質之王
    candidate_models = ['gemini-1.5-pro-002', 'gemini-flash-latest']

    for model_name in candidate_models:
        try:
            logger.info(f"🧪 嘗試使用模型: {model_name} 進行撰寫...")
            
            response = client.models.generate_content(
                model=model_name, 
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3
                )
            )
            logger.info(f"✨ 成功使用 {model_name} 完成報告！")
            return response.text
            
        except Exception as e:
            logger.warning(f"⚠️ 模型 {model_name} 執行失敗 (可能是額度不足或不支援): {e}")
            logger.info("🔄 正在切換至下一個備援模型...")
            continue # 繼續迴圈，試下一個模型

    logger.error("❌ 所有模型皆嘗試失敗，無法生成報告。")
    return None

    try:
        # 使用 2.0-flash 
        response = client.models.generate_content(
            model='gemini-flash-latest', 
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3
           )
        )
        return response.text
    except Exception as e:
        logger.error(f"❌ Gemini 生成失敗: {e}")
        return None

def send_line_push(message):
    logger.info("🚀 正在發送 LINE 訊息...")
    configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
    try:
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            push_message_request = PushMessageRequest(
                to=LINE_USER_ID,
                messages=[TextMessage(text=message)]
            )
            line_bot_api.push_message(push_message_request)
            logger.info("✅ LINE 訊息發送成功！")
    except Exception as e:
        logger.error(f"❌ LINE 發送失敗: {e}")

def main():
    today = get_target_date()
    news = search_news(today)
    
    if not news:
        logger.warning("📭 今天沒有足夠的新聞，跳過。")
        return

    summary = generate_summary(news, today)
    
    if summary:
        print("\n" + "="*30)
        print(summary)
        print("="*30 + "\n")
        send_line_push(summary)


if __name__ == "__main__":
    main()