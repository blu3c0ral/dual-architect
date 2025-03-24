from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Now you can access them with os.environ
api_key = os.environ.get("LLM_API_KEY")
