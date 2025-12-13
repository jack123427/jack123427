# 台灣股市預測專案 (Proof of Concept)

這是一個使用台灣證券交易所 (TWSE) 的公開 API 來預測台灣股市走向的概念性驗證專案。

## 專案功能

*   **資料擷取**: 從 TWSE OpenAPI 獲取上市公司的基本資料、每日股價、財務報表、本益比、股價淨值比、融資融券等資料。
*   **市值計算**: 計算所有上市公司的市值，並篩選出市值前一百大的公司。
*   **模型訓練與預測**: 使用 XGBoost 模型，結合時間序列特徵和延遲特徵，對指定股票的未來 10 天收盤價進行預測。
*   **模型回測**: 提供一個簡單的回測機制，使用平均絕對百分比誤差 (MAPE) 來評估模型的準確度。
*   **API 服務**: 提供一個基於 Flask 的 API 服務，讓使用者可以透過 HTTP 請求來獲取指定股票的預測結果。

## 檔案結構與功能

```
.
├── src
│   ├── data_fetcher.py   # 負責從 TWSE API 獲取所有需要的資料。
│   ├── model.py          # 包含了 XGBoost 模型的訓練、預測與回測邏輯。
│   ├── app.py            # Flask API 服務，提供 /predict 端點。
│   └── main.py           # 主要的執行腳本，串連所有功能，對市值前五大公司進行分析。
├── tests
│   ├── test_data_fetcher.py # data_fetcher.py 的單元測試。
│   └── test_model.py     # model.py 的單元測試。
├── .gitignore            # 忽略不需要加入版本控制的檔案。
└── requirements.txt      # 專案所需的核心 Python 相依套件。
```

## 安裝與執行

### 1. 環境準備

強烈建議使用 [Anaconda](https://www.anaconda.com/products/distribution) 來管理您的 Python 環境，這可以大幅簡化安裝 `xgboost` GPU 版本的流程。

建議使用 Python 3.8 或更新的版本。

### 2. 安裝相依套件

#### a) 使用 Anaconda (建議)

1.  **建立新的 conda 環境**：
    ```bash
    conda create --name stock-prediction python=3.9
    conda activate stock-prediction
    ```
2.  **安裝核心套件**：
    ```bash
    pip install -r requirements.txt
    ```
3.  **安裝 XGBoost** (二選一)：
    *   **CPU 版本**:
        ```bash
        conda install -c conda-forge xgboost
        ```
    *   **GPU 版本**: `conda` 會自動處理 CUDA toolkit 的相依性，是目前最建議的 GPU 版本安裝方式。
        ```bash
        conda install -c conda-forge py-xgboost-gpu
        ```

#### b) 使用 pip (不建議用於 GPU 版本)

如果您不使用 Anaconda，您也可以使用 `pip` 來安裝。

```bash
# 步驟 1: 安裝核心套件
pip install -r requirements.txt

# 步驟 2: 安裝 XGBoost (CPU 版本)
pip install xgboost
```

### 3. 執行方式

#### a) 執行主要分析流程

您可以直接執行 `main.py` 來啟動對市值前五大公司的完整分析流程 (資料獲取 -> 訓練 -> 回測 -> 預測)。

```bash
# Linux / macOS
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 src/main.py

# Windows PowerShell
$env:PYTHONPATH += ";$(pwd)/src"
python3 src/main.py
```

#### b) 啟動 API 服務

您可以執行 `app.py` 來啟動一個 API 伺服器。

```bash
python3 src/app.py
```

伺服器啟動後，您可以使用 `curl` 或其他工具來測試 API。例如，要預測台積電 (2330) 的股價：

```bash
curl "http://127.0.0.1:5000/predict?stock_code=2330"
```

## Visual Studio Code (VSCode) 使用者指南

1.  **選擇 Python 直譯器**：
    *   打開 VSCode 的命令面板 (View -> Command Palette... 或 `Ctrl+Shift+P`)。
    *   輸入 "Python: Select Interpreter"。
    *   選擇您希望使用的 Python 環境 (例如，您用 Anaconda 建立的 `stock-prediction` 環境)。
2.  **設定 `launch.json` 以方便偵錯**：
    *   切換到 "Run and Debug" 分頁。
    *   點擊 "create a launch.json file"，並選擇 "Python"。
    *   VSCode 會產生一個 `launch.json` 檔案。將其內容修改如下，主要是加入了 `"env": {"PYTHONPATH": "${workspaceFolder}/src"}` 這個設定，這樣 VSCode 在偵錯時才能正確地找到 `src` 目錄下的模組。
    ```json
    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Main",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/src/main.py",
                "console": "integratedTerminal",
                "env": {
                    "PYTHONPATH": "${workspaceFolder}/src"
                }
            },
            {
                "name": "Python: Flask API",
                "type": "python",
                "request": "launch",
                "module": "flask",
                "env": {
                    "FLASK_APP": "src/app.py",
                    "PYTHONPATH": "${workspaceFolder}/src"
                },
                "args": [
                    "run",
                    "--no-debugger"
                ],
                "jinja": true
            }
        ]
    }
    ```
3.  **開始偵錯**：
    *   現在您可以在 "Run and Debug" 分頁的下拉選單中，選擇 "Python: Main" 來執行 `main.py`，或選擇 "Python: Flask API" 來啟動 API 伺服器，並可以設定中斷點進行偵錯。
