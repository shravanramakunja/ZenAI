#  Deployment Instructions

## GitHub Secrets Setup

Before running the GitHub Actions workflow, you need to add these secrets to your GitHub repository:

### Required Secrets:

1. **HF_TOKEN** (Already added )
   - Your Hugging Face access token
   - Used for authenticating with Hugging Face Spaces

2. **GOOGLE_API_KEY** 
   - Your Google Gemini API key: 
   - Used for the Gemini 2.0 Flash API

### How to Add GitHub Secrets:

1. Go to your GitHub repository
2. Click on **Settings** tab
3. Navigate to **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Add the following secrets:

```
Name: GOOGLE_API_KEY
Value: YOUR API KEy
```

## Deployment Process

### Automatic Deployment (Recommended)

1. **Push to main branch** - The workflow triggers automatically
2. **Manual deployment** - Go to Actions tab → "Deploy to Hugging Face Spaces" → "Run workflow"

### What the Deployment Does:

1. ✅ Creates/updates Hugging Face Space: `shravanramakunja/ZenAI`
2. ✅ Uploads all necessary files (app.py, src/, data/, requirements.txt)
3. ✅ Configures environment variables
4. ✅ Sets up Streamlit SDK configuration
5. ✅ Builds and deploys your ZenAI Medical Chatbot

### Expected Deployment URL:

 **https://huggingface.co/spaces/shravanramakunja/ZenAI**

### Deployment Timeline:

- **GitHub Actions**: ~2-3 minutes
- **Hugging Face Build**: ~5-10 minutes
- **Total Time**: ~7-13 minutes

## Troubleshooting

### If deployment fails:

1. **Check GitHub Actions logs** in the Actions tab
2. **Verify secrets** are correctly set
3. **Check Hugging Face token** permissions
4. **Ensure repository name** matches in workflow file

### Common Issues:

- **Invalid HF_TOKEN**: Make sure token has write access to spaces
- **API Key Error**: Verify Google API key is correctly added to secrets
- **File Upload Error**: Check if all required files exist in repository

## Post-Deployment

After successful deployment:

1.  Visit your Hugging Face Space
2.  Test the medical chatbot functionality
3.  Share your deployed application!

---


