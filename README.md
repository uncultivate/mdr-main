# Magic Squares Detection and Revelation (MDR)

An interactive web-based game where a user (or AI) explores a grid to find magic squares - arrangements of numbers where each row, column, and main diagonal add up to the same sum.

## Features

- Interactive grid exploration with mouse hover scaling effects
- Multiple search strategies including:
  - Exploration Priority
  - Random Walk
  - Spiral Search
  - Pattern Detection
- AI mode that automatically explores the grid
- Custom search strategy builder

## Local Development

### Backend (Flask)

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Flask backend:
   ```
   python app.py
   ```
   The backend will run on http://localhost:5000

### Frontend (Next.js)

1. Navigate to the frontend directory:
   ```
   cd mdr
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Run the development server:
   ```
   npm run dev
   ```
   The frontend will run on http://localhost:3000

## Deployment Options

### Deploy to Production

#### Backend Options

1. **Render**
   - Connect your GitHub repository
   - Select Python as the runtime
   - Set the build command: `pip install -r requirements.txt`
   - Set the start command: `gunicorn app:app`
   - Set the environment variables:
     - `PRODUCTION=True`
     - `DEBUG=False`

2. **Heroku**
   - Create a Heroku account and install the Heroku CLI
   - Run:
     ```
     heroku create your-app-name
     git push heroku main
     ```
   - Heroku will automatically use the Procfile to run your app

#### Frontend Options

1. **Vercel (Recommended for Next.js)**
   - Sign up at vercel.com and connect your GitHub repository
   - Set the environment variable:
     - `NEXT_PUBLIC_API_URL=https://your-backend-url.com`
   - Deploy with:
     ```
     cd mdr
     vercel
     ```

2. **Netlify**
   - Connect your GitHub repository
   - Set build command: `cd mdr && npm install && npm run build`
   - Set publish directory: `mdr/.next`
   - Set environment variables the same as with Vercel

## Important Deployment Notes

1. Make sure to update the CORS settings in app.py with your production frontend URL
2. Update the API_BASE_URL in the frontend to point to your deployed backend
3. Set environment variables appropriately for both frontend and backend 