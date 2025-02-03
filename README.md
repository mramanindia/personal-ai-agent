# personal-ai-agent


This project provides a graphical user interface (GUI) for selecting a PDF file, entering an OpenAI API key, and running a text extraction and query process using FAISS and OpenAI. It allows you to interact with data from a PDF document and get relevant answers based on your queries.

## Features:
- Select a PDF file for text extraction.
- Input your OpenAI API key to access the OpenAI service.
- The tool extracts text from the PDF, processes it using FAISS, and retrieves relevant chunks based on a query.
- Outputs the answer from OpenAI based on the selected PDF's content.

## Requirements

Ensure you have the following Python dependencies installed:

- `tkinter` (comes with Python by default)
- `faiss-cpu` (or `faiss-gpu` if you're using GPU support)
- `openai` (for OpenAI API access)
- `PyPDF2` (for PDF reading and extraction)

### Installation Steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/pdf-openai-query-tool.git
   cd pdf-openai-query-tool


2. **Create a virtual environment:**
   - For **Windows**:
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - For **macOS/Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   Start the GUI application by running:
   ```bash
   python main.py
   ```

### Using the Application:

1. **Select a PDF File**: Click the "Browse" button to select the PDF file you want to process.
2. **Enter OpenAI API Key**: Input your OpenAI API key in the provided text field.
3. **Run the Process**: Click the "Run" button to extract data from the PDF, process it using FAISS, and get an answer based on your query.

### Troubleshooting:

- **Tkinter Issues**: If `tkinter` is not installed, follow your OS-specific instructions to install it (for example, on Linux, use `sudo apt-get install python3-tk`).
- **macOS Version Issue**: If you get an error related to macOS version compatibility, ensure your macOS is updated or install an older Python version compatible with your macOS version.

## License

This project is open-source and available under the MIT License.

```

### Notes:
- Replace `https://github.com/your-username/pdf-openai-query-tool.git` with the actual URL of your repository.
- The installation steps include creating and activating a virtual environment and installing dependencies from `requirements.txt`.
- The usage section explains how to interact with the application.
