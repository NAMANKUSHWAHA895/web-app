import os
import google.generativeai as genai
from flask import Flask, render_template, request
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- Configure Google AI ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    # If the API key is not found, raise an error or handle it appropriately
    # For this example, we'll print a message and exit, but in a real app,
    # you might want to show an error page or log the issue.
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    print("Make sure you have a .env file with GOOGLE_API_KEY=YOUR_API_KEY")
    # For simplicity, we'll let it potentially fail later if None,
    # but checking early is better practice.
    # exit(1) # Or raise an exception

try:
    genai.configure(api_key=api_key)
    # Choose a model - 'gemini-1.5-flash' is a good starting point (fast and capable)
    # Other options: 'gemini-1.0-pro', 'gemini-1.5-pro-latest'
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Google AI Model initialized successfully.")
except Exception as e:
    print(f"Error initializing Google AI Model: {e}")
    # Handle initialization failure gracefully in a real app
    model = None # Set model to None if initialization fails


# --- Safety Settings (Optional but Recommended) ---
# Adjust thresholds as needed: BLOCK_NONE, BLOCK_LOW_AND_ABOVE, BLOCK_MEDIUM_AND_ABOVE, BLOCK_ONLY_HIGH
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

@app.route('/', methods=['GET', 'POST'])
def index():
    query = None
    result = None
    error = None

    if request.method == 'POST':
        query = request.form.get('query')

        if not query:
            error = "Please enter a query."
        elif not model:
             error = "AI Model is not available due to configuration error."
        else:
            try:
                # --- Send query to Google AI ---
                print(f"Sending query to Gemini: '{query}'") # Log the query being sent
                response = model.generate_content(
                    query,
                    safety_settings=safety_settings
                    # You can add generation_config here for more control (temperature, max_tokens, etc.)
                    # generation_config=genai.types.GenerationConfig(temperature=0.7)
                )
                print("Received response from Gemini.") # Log successful response

                # --- Process the response ---
                # Check if the response was blocked due to safety settings
                if response.prompt_feedback.block_reason:
                    error = f"Request blocked due to: {response.prompt_feedback.block_reason.name}. Try rephrasing your query."
                    print(f"Query blocked: {response.prompt_feedback.block_reason.name}") # Log block reason
                else:
                    # Check if the response candidate exists and has content parts
                    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                        result = response.text # .text safely extracts the text content
                        print("Successfully extracted text from response.")
                    else:
                        # Handle cases where generation might be empty or stopped for other reasons
                        finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                        error = f"AI did not generate a response. Reason: {finish_reason.name if hasattr(finish_reason, 'name') else finish_reason}"
                        print(f"Empty or incomplete response. Finish Reason: {finish_reason}") # Log this case

            except Exception as e:
                # Catch potential API errors or other issues during generation
                error = f"An error occurred while contacting the AI: {str(e)}"
                print(f"Error during AI generation: {e}") # Log the specific error

    # Render the HTML template, passing the variables
    return render_template('index.html', query=query, result=result, error=error)

if __name__ == '__main__':
    # Runs the Flask development server
    # Debug=True enables auto-reloading and detailed error pages (disable for production)
    app.run(debug=False, host='0.0.0.0')