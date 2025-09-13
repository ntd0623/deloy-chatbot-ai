import chatbotai

if __name__ == "__main__":
    chatbotai.train_from_excel(chatbotai.EXCEL_FILE)
    print("✅ Dữ liệu từ Excel đã được import vào data.json và nạp vào bộ nhớ!")
