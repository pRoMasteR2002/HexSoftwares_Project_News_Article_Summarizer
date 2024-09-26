
# 🌟 News Article Summarizer 🌟

## 📖 Overview
Welcome to the **News Article Summarizer**! This innovative tool harnesses the power of **Natural Language Processing (NLP)** to transform lengthy news articles into concise summaries. By utilizing state-of-the-art techniques from the **Hugging Face Transformers** library, this application not only saves time for readers but also enhances the accessibility of information.

---

## 🚀 How to Get Started

### ⚙️ Prerequisites
Before diving into the application, ensure you have the following libraries installed:

- **Streamlit**: For building the web application.
- **Pandas**: For data manipulation and analysis.
- **Python-docx**: For processing DOCX files.
- **Transformers**: For NLP models.
- **Torch**: For running the models.
- **TensorFlow**: For deep learning capabilities.
- **Python-dotenv**: For managing environment variables.
- **PSUtil**: For monitoring system resources.

You can easily install these libraries using pip:

```bash
pip install streamlit pandas python-docx transformers torch tensorflow python-dotenv psutil
```

### 🛠️ Setting Up Environment Variables on Windows

#### Paths to Add
To ensure everything runs smoothly, you’ll need to add the following paths to your environment variables:

1. **Python Installation Path**: 
   ```
   C:\Users\<YourUserName>\AppData\Local\Programs\Python\Python312\
   ```
   
2. **Scripts Folder Path**: 
   ```
   C:\Users\<YourUserName>\AppData\Local\Programs\Python\Python312\Scripts\
   ```

#### Steps to Add to PATH
1. **Open Environment Variables**:
   - Right-click on **This PC** or **My Computer** on your desktop or in File Explorer, and choose **Properties**.
   - Click on **Advanced system settings**.
   - In the **System Properties** window, click on the **Environment Variables** button.

2. **Edit the PATH Variable**:
   - In the **Environment Variables** window, scroll under **System variables** to find **Path**. Select it and click **Edit**.
   - In the **Edit Environment Variables** window, click **New** and add:
     - `C:\Users\<YourUserName>\AppData\Local\Programs\Python\Python312\`
     - `C:\Users\<YourUserName>\AppData\Local\Programs\Python\Python312\Scripts\`

3. **Save and Restart**:
   - Click **OK** to close all dialog boxes.
   - Restart your terminal or PowerShell to apply the changes.

### 💻 Running the Application
To launch the summarization tool, navigate to the project directory and run the following command:

```bash
python -m streamlit run summarizer.py
```

Alternatively, you can use:

```bash
streamlit run summarizer.py
```

---

## 🛠️ How It Works
The **News Article Summarizer** operates by allowing users to input text directly or upload files (CSV, TXT, or DOCX) containing articles. The application intelligently processes the input using a sophisticated summarization model, providing concise summaries that are easy to digest. Here's how it works:

1. **Input Options**: Users can either paste text or upload a file for summarization.
2. **Text Processing**: The application analyzes the input and applies the summarization model.
3. **Output Generation**: The concise summary is displayed along with additional details, such as the author's name and word count.

### 🔍 Under the Hood
The summarization process utilizes an advanced model from Hugging Face, allowing for both **Extractive** and **Abstractive** summarization techniques. The model is fine-tuned to understand context and generate coherent summaries that maintain the original meaning of the text. This ensures that users receive concise yet informative outputs.

---

## 🌟 Key Features
- **Dynamic Text Summarization**: Quickly transforms extensive articles into concise summaries.
- **Versatile File Uploads**: Supports CSV, TXT, and DOCX file formats for easy input.
- **User-Friendly Interface**: Built with **Streamlit** for an intuitive user experience.
- **Performance Metrics**: Displays memory usage and summarization time to monitor efficiency.
- **Customizable Output**: Users can choose the number of articles to summarize and specific articles based on indices.

---

## ⚠️ Error Handling
Should you encounter issues while running the application, please consider the following:
- Verify that all required libraries are installed.
- Ensure your environment variables are correctly set.
- Remember, the input text must exceed 100 words for effective summarization.
- For any additional support, refer to the project's GitHub page: [GitHub Repository](https://github.com/pRoMasteR2002/HexSoftwares_Project_News_Article_Summarizer/tree/main).

---

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements
A special thanks to **HexSoftwares** for the opportunity to develop this project and to the open-source community, particularly **Hugging Face**, for their invaluable contributions to the field of NLP.

---
