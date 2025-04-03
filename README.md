## Using .env Files

Langvio supports loading API keys from a `.env` file. This is a convenient way to manage your credentials without hardcoding them in your code.

1. Create a `.env` file in your project directory using the provided template:

```bash
# Copy the template
cp .env.template .env

# Edit the file with your actual API keys
nano .env  # or use your preferred editor
```

2. Add your actual API keys to the `.env` file:

```
GOOGLE_API_KEY=your_actual_google_api_key_here
OPENAI_API_KEY=your_actual_openai_api_key_here
# Add other API keys as needed
```

3. Langvio will automatically load these environment variables when imported:

```python
import langvio

# No need to manually set environment variables!
# API keys are automatically loaded from .env file

pipeline = langvio.create_pipeline()
# ... use the pipeline as usual
```

**Important Note**: Make sure to add `.env` to your `.gitignore` file to prevent accidentally committing your API keys to version control!

```bash
echo ".env" >> .gitignore
```