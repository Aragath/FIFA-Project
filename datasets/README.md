# Datasets
## raw:
爬蟲下來的data

## pre-processed:
1. 日期格式改為datetime64[ns](i.e. 2022-06-21)
2. 移除完全重複的row
3. 移除完全空白的row（不知道產生的原因，但前後的player stat都沒有少）
4. 移除名稱為FakePlayer的row（此次raw中沒出現）
 
