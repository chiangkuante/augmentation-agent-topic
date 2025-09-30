from sec_edgar_downloader import Downloader
import os
from datetime import datetime


def download_sec_filings():
  # --- 步驟 1: 初始化 Downloader ---
  download_folder = os.path.join("data")
  dl = Downloader("NPUST MIS", "chiangkuante@gmail.com", download_folder)

  # --- 步驟 2: 設定要爬取的公司與文件類型 ---
  # 設定美國前20大企業的股票代碼 (2018-2019年期間的市值排名)
  tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "BRK-B", "META", "JNJ", "XOM", "JPM", "V",
            "PG", "UNH", "HD", "BAC", "MA", "DIS", "INTC", "VZ", "CSCO", "PFE"] 
  # 設定要爬取的文件類型。'10-K' 就是年報的代碼。
  # 你也可以下載季報 '10-Q' 或重大事件報告 '8-K' 等。
  filing_type = "10-K"

  # --- 步驟 3: 設定日期範圍與下載數量 ---
  # 設定日期範圍為2018-2019年
  start_date = "2018-01-01"
  end_date = "2019-12-31"

  # --- 步驟 4: 執行下載 ---
  # 使用 for 迴圈遍歷所有指定的公司
  for ticker in tickers:
      try:
          print(f"正在處理 {ticker}...")
          # get 方法是核心下載函數
          # 參數說明:
          # 1. filing_type: 文件類型 ('10-K')
          # 2. ticker: 公司股票代碼 (例如 'AAPL')
          # 3. after: 開始日期 (可選)
          # 4. before: 結束日期 (可選)
          # 5. download_details: 是否下載包含詳細資訊的 HTML 文件，設為 True。
          num_filings_downloaded = dl.get(filing_type,
                                          ticker,
                                          after=start_date,
                                          before=end_date,
                                          download_details=True)

          if num_filings_downloaded < 2:
              print(f"⚠️  {ticker} 只下載到 {num_filings_downloaded} 份文件 (預期2份)")
              print(f"   可能原因: 該公司在指定期間內沒有提交完整的年報，或年報提交日期在範圍外")
          else:
              print(f"✅ 成功下載 {ticker} 的 {num_filings_downloaded} 份 2018-2019年 {filing_type} 年報。")

      except Exception as e:
          print(f"❌ 下載 {ticker} 的文件時發生錯誤: {e}")
          print(f"   建議: 檢查股票代碼是否正確，或該公司是否在SEC資料庫中")

  original_folder_path = os.path.join("data", "sec-edgar-filings")
  new_folder_path = os.path.join("data", "sec_data")
  os.rename(original_folder_path, new_folder_path)

  print("\n" + "="*60)
  print("美國前20大企業2018-2019年10-K年報下載任務已完成。")
  print(f"下載時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
  print("\n說明:")
  print("• 每家公司預期下載2份文件 (2018年和2019年的10-K年報)")
  print("• 如果某家公司只有1份文件，可能是因為:")
  print("  - 公司的會計年度結束日期導致年報提交時間落在範圍外")
  print("  - 公司在該期間進行了重組或其他企業行為")
  print("  - SEC資料庫中的資料完整性問題")
  print("="*60)

