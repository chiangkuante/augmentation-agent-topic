import nltk
from pathlib import Path
import os
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import tiktoken
import re
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import time
import pandas as pd



def process_sec_filings(config):
  nltk.download('punkt_tab')
  sec_data_dir = Path("data/sec_data")
  all_chunks_data = []
  for ticker in os.listdir(sec_data_dir):
    ticker_path = os.path.join(sec_data_dir, ticker, '10-K')
    if not os.path.isdir(ticker_path):
      continue
    print(f"--- 正在處理公司: {ticker} ---")

    for filing_folder in os.listdir(ticker_path):
      filing_path = os.path.join(ticker_path, filing_folder)
      html_path = os.path.join(filing_path, 'primary-document.html')

      if os.path.exists(html_path):
        print(f"  處理文件: {filing_folder}...")
        with open(html_path, 'r', encoding='utf-8') as f:
          html_content = f.read()

        # 步驟 1: 提取全文
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html_content, 'lxml')
        # 移除不必要的標籤，如表格、樣式、腳本等
        for tag in soup(["style", "script", "table"]):
          tag.decompose()
        # 提取所有文字
        full_text = soup.get_text(separator='\n', strip=True)
        # 移除多餘的換行符和空白
        full_text = re.sub(r'\n{3,}', '\n\n', full_text) # 將三個以上的換行符合併為兩個
        full_text = re.sub(r'\s{2,}', ' ', full_text)   # 將兩個以上的空白符合併為一個

        # 步驟 2: 句子過濾
        sentences = nltk.sent_tokenize(full_text)
        filtered_sentences = []
        # 定義樣板化法律聲明的關鍵詞
        boilerplate_keywords = [
          "forward-looking statements", "safe harbor", 
          "risks and uncertainties", "pursuant to section",
          "in witness whereof"
        ]
        for sent in sentences:
          # 1. 過濾過短的句子 (例如，少於 10 個詞)
          if len(sent.split()) < 10:
            continue
          # 2. 過濾樣板化法律聲明 (檢查是否包含關鍵詞)
          # str.lower() 將句子轉為小寫以利比對
          if any(keyword in sent.lower() for keyword in boilerplate_keywords):
            continue
          filtered_sentences.append(sent)
        sentences = filtered_sentences

        # 步驟 3: 語義分塊
        max_tokens_per_chunk = int(config.get("MAX_TOKENS_PER_CHUNK"))
        # 初始化 tokenizer 以便計算 token 數量
        tokenizer = tiktoken.get_encoding("cl100k_base")

        # --- 第一階段：基於結構的初步分塊 (按段落和句子) ---
        full_clean_text = "\n".join(sentences)
        # 嘗試多種分割方式來確保分塊
        paragraphs = []

        # 按雙換行分割
        initial_split = full_clean_text.split('\n\n')

        for para in initial_split:
          para_token_count = len(tokenizer.encode(para))

          # 如果單個段落超過限制，使用滑動窗口分割
          if para_token_count > max_tokens_per_chunk:
            para_sentences = para.split('\n')

            # 滑動窗口參數
            window_size = 15  # 每個窗口包含的句子數
            overlap_size = 3  # 重疊的句子數，保持語義連續性

            for i in range(0, len(para_sentences), window_size - overlap_size):
              end_idx = min(i + window_size, len(para_sentences))
              chunk_sentences = para_sentences[i:end_idx]

              # 檢查這個窗口是否還是太大
              chunk_text = '\n'.join(chunk_sentences)
              chunk_tokens = len(tokenizer.encode(chunk_text))

              if chunk_tokens > max_tokens_per_chunk:
                # 如果窗口還是太大，縮小窗口大小
                smaller_window = window_size // 2
                for j in range(i, end_idx, smaller_window):
                  small_end = min(j + smaller_window, len(para_sentences))
                  small_chunk = para_sentences[j:small_end]
                  paragraphs.append('\n'.join(small_chunk))
              else:
                paragraphs.append(chunk_text)

              # 如果已經到達末尾，退出循環
              if end_idx >= len(para_sentences):
                break
          else:
            paragraphs.append(para)
        
        # --- 第二階段：合併段落成中間區塊 ---
        mega_chunks = []
        current_chunk = ""
        current_token_count = 0
        
        for para in paragraphs:
          para_token_count = len(tokenizer.encode(para))
          
          if current_token_count + para_token_count > max_tokens_per_chunk:
            # 如果加入這個段落會超過上限，就將當前的 chunk 存起來
            if current_chunk:
              mega_chunks.append(current_chunk)
            # 開始一個新的 chunk
            current_chunk = para
            current_token_count = para_token_count
          else:
            # 如果未超過上限，就繼續添加
            current_chunk += f"\n\n{para}"
            current_token_count += para_token_count
            
        # 儲存最後一個 chunk
        if current_chunk:
          mega_chunks.append(current_chunk)
          
        print(f"    > 已將全文初步切分成 {len(mega_chunks)} 個中間區塊。")

        # --- 第三階段：對每個中間區塊進行語義分塊 ---
        final_semantic_chunks = []
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        text_splitter = SemanticChunker(
          embeddings, 
          breakpoint_threshold_type="percentile",
          breakpoint_threshold_amount=95
        )
        for i, mega_chunk in enumerate(mega_chunks):
          print(f"    > 正在處理中間區塊 {i+1}/{len(mega_chunks)}...")
          chunk_token_count = len(tokenizer.encode(mega_chunk))
          print(f"    > 區塊 token 數: {chunk_token_count}")

          # 對每個中間區塊調用 create_documents
          chunks = text_splitter.create_documents([mega_chunk])
          final_semantic_chunks.extend([doc.page_content for doc in chunks])
          print(f"    > 語義分割完成，生成 {len(chunks)} 個區塊")

          # 添加延遲以符合 API 速率限制 (Tier 3: 每分鐘 5000 requests)
          if i < len(mega_chunks) - 1:  # 最後一個區塊不需要延遲
            time.sleep(0.5)  # 0.5 秒延遲
        final_chunks = final_semantic_chunks

        year = filing_folder.split('-')[1]
        for i, chunk in enumerate(final_chunks):
          all_chunks_data.append({
            'ticker': ticker,
            'year': f"20{year}",
            'chunk_id': f"{filing_folder}_{i}",
            'text': chunk
          })
        print(f"    > 最終成功生成 {len(final_chunks)} 個語義文本塊。")

  # 步驟 4: 儲存最終語料庫
  corpus_df = pd.DataFrame(all_chunks_data)
  DATA_CSV_PATH = config.get("DATA_CSV_PATH")
  corpus_df.to_csv(DATA_CSV_PATH, index=False, encoding='utf-8-sig')
  print(f"\n--- 處理完成！最終語料庫已儲存至 {DATA_CSV_PATH} ---")

