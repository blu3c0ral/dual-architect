from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


# Define global providers keys
generator_provider = os.getenv("GENERATOR_PROVIDER")
validator_provider = os.getenv("VALIDATOR_PROVIDER")
test_provider = os.getenv("TEST_PROVIDER")
