# 🚀 Deployment Guide - StrokeCare AI

This guide covers deploying both the Flask and Streamlit versions of the StrokeCare AI application.

## 📋 Prerequisites

- Python 3.8 or higher
- The two model files:
  - `stroke_binary_model.pkl`
  - `stroke_probability_model.pkl`

## 🔧 Local Development Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Model Files

Ensure both model files are in the project root:
- `stroke_binary_model.pkl`
- `stroke_probability_model.pkl`

### 3. Run Flask App

```bash
python app.py
```

The app will be available at `http://localhost:5000`

### 4. Run Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended for Streamlit)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `streamlit_app.py`
   - Click "Deploy"

3. **Add Model Files**
   - Upload `stroke_binary_model.pkl` and `stroke_probability_model.pkl` to your GitHub repo
   - Or use Streamlit Cloud's file upload feature

### Option 2: Heroku (Flask App)

1. **Install Heroku CLI**
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Create Procfile**
   ```
   web: gunicorn app:app
   ```

3. **Create runtime.txt**
   ```
   python-3.11.0
   ```

4. **Deploy**
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

5. **Add Model Files**
   - Use Heroku's file system (temporary) or
   - Store models in cloud storage (S3, etc.) and load at runtime

### Option 3: Railway (Both Apps)

1. **Install Railway CLI**
   ```bash
   npm i -g @railway/cli
   ```

2. **Deploy**
   ```bash
   railway login
   railway init
   railway up
   ```

3. **Configure**
   - For Flask: Set start command to `gunicorn app:app`
   - For Streamlit: Set start command to `streamlit run streamlit_app.py --server.port $PORT`

### Option 4: Docker (Both Apps)

1. **Create Dockerfile for Flask**
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 5000
   CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
   ```

2. **Create Dockerfile for Streamlit**
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

3. **Build and Run**
   ```bash
   # Flask
   docker build -t stroke-flask -f Dockerfile.flask .
   docker run -p 5000:5000 stroke-flask
   
   # Streamlit
   docker build -t stroke-streamlit -f Dockerfile.streamlit .
   docker run -p 8501:8501 stroke-streamlit
   ```

### Option 5: AWS/GCP/Azure

#### AWS (Elastic Beanstalk)

1. **Install EB CLI**
   ```bash
   pip install awsebcli
   ```

2. **Initialize**
   ```bash
   eb init -p python-3.11 stroke-app
   eb create stroke-env
   eb deploy
   ```

#### Google Cloud Platform (Cloud Run)

1. **Create Dockerfile** (see Option 4)

2. **Deploy**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/stroke-app
   gcloud run deploy --image gcr.io/PROJECT_ID/stroke-app --platform managed
   ```

#### Azure (App Service)

1. **Install Azure CLI**
   ```bash
   az login
   az webapp up --name stroke-app --runtime "PYTHON:3.11"
   ```

## 📝 Environment Variables

For production, consider setting:

```bash
FLASK_ENV=production
STREAMLIT_SERVER_PORT=8501
OPENROUTER_API_KEY=your_key_here  # If using AI insights
```

## 🔒 Security Considerations

1. **API Keys**: Never commit API keys. Use environment variables or secrets management.

2. **Model Files**: 
   - Keep model files secure
   - Consider versioning models
   - Use cloud storage for large models

3. **HTTPS**: Always use HTTPS in production

4. **Rate Limiting**: Consider adding rate limiting for production

## 🐛 Troubleshooting

### Models Not Loading

- Verify model files are in the correct location
- Check file permissions
- Ensure pickle version compatibility

### Port Issues

- Flask default: 5000
- Streamlit default: 8501
- Use environment variable `PORT` for cloud platforms

### Dependencies Issues

- Use virtual environment
- Pin dependency versions in `requirements.txt`
- Test locally before deploying

## 📊 Monitoring

Consider adding:
- Application logging
- Error tracking (Sentry, etc.)
- Performance monitoring
- Usage analytics

## 🔄 Updates

To update the deployed app:

1. Make changes locally
2. Test thoroughly
3. Commit and push to repository
4. Platform will auto-deploy (if configured) or manually deploy

## 📞 Support

For issues or questions:
- Check application logs
- Review error messages
- Verify model files are present
- Test locally first

---

**Note**: Always test thoroughly in a staging environment before deploying to production!

