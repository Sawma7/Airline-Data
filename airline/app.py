import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages

PLOTS_PATH = 'static/plots'
UPLOADS_PATH = 'uploads'

# Ensure plots folder and uploads folder exist
os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(UPLOADS_PATH, exist_ok=True)

DATABASE = 'users.db'

def init_db():
    """Initialize the user database if it doesn't already exist."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fullname TEXT NOT NULL,
                        email TEXT NOT NULL UNIQUE,
                        username TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()  # Initialize the database

# Save plot
def save_plot(fig, filename):
    fig.savefig(os.path.join(PLOTS_PATH, filename), bbox_inches='tight')
    plt.clf()

# Generate EDA visualizations
def generate_eda(df):
    # 1. Gender distribution
    sns.countplot(data=df, x='Gender')
    plt.title('Gender Distribution')
    save_plot(plt, 'gender_distribution.png')

    # 2. Age distribution
    sns.histplot(data=df, x='Age', kde=True)
    plt.title('Age Distribution')
    save_plot(plt, 'age_distribution.png')

    # 3. Flight status count
    sns.countplot(data=df, x='Flight Status')
    plt.title('Flight Status')
    save_plot(plt, 'flight_status.png')

    # 4. Flight status by continent
    sns.countplot(data=df, x='Airport Continent', hue='Flight Status')
    plt.title('Flight Status by Continent')
    plt.xticks(rotation=45)
    save_plot(plt, 'status_by_continent.png')

    # 5. Top 5 nationalities
    top_nations = df['Nationality'].value_counts().nlargest(5)
    sns.barplot(x=top_nations.index, y=top_nations.values)
    plt.title('Top 5 Passenger Nationalities')
    save_plot(plt, 'top_nationalities.png')

    # 6. Top 10 Departure Airports
    top_airports = df['Airport Name'].value_counts().nlargest(10)
    sns.barplot(x=top_airports.index, y=top_airports.values)
    plt.title('Top 10 Departure Airports')
    plt.xticks(rotation=45)
    save_plot(plt, 'top_departure_airports.png')

    # 7. Flight Status by Gender
    sns.countplot(data=df, x='Gender', hue='Flight Status')
    plt.title('Flight Status by Gender')
    save_plot(plt, 'status_by_gender.png')

    # 8. Average Age per Flight Status
    sns.barplot(data=df, x='Flight Status', y='Age')
    plt.title('Average Age by Flight Status')
    save_plot(plt, 'avg_age_by_status.png')

    # 9. Flight Count by Airport Country Code
    top_codes = df['Airport Country Code'].value_counts().nlargest(10)
    sns.barplot(x=top_codes.index, y=top_codes.values)
    plt.title('Top 10 Country Codes')
    save_plot(plt, 'top_country_codes.png')

    # 10. Heatmap of Gender vs Continent
    heatmap_data = pd.crosstab(df['Gender'], df['Airport Continent'])
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues')
    plt.title('Flight Count by Gender and Continent')
    save_plot(plt, 'heatmap_gender_continent.png')

    # 11. Top 10 Pilots
    top_pilots = df['Pilot Name'].value_counts().nlargest(10)
    sns.barplot(x=top_pilots.index, y=top_pilots.values)
    plt.title('Top 10 Pilots by Flight Count')
    plt.xticks(rotation=45)
    save_plot(plt, 'top_pilots.png')

    # 12. Monthly Flight Trend
    df['Month'] = df['Departure Date'].dt.to_period('M')
    monthly_flights = df.groupby('Month').size()
    monthly_flights.plot(marker='o')
    plt.title('Monthly Flight Trend')
    plt.xlabel('Month')
    plt.ylabel('Number of Flights')
    save_plot(plt, 'monthly_flight_trend.png')

    # 13. Weekday Flight Status
    df['Weekday'] = df['Departure Date'].dt.day_name()
    sns.countplot(data=df, x='Weekday', hue='Flight Status', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.title('Flight Status by Day of the Week')
    plt.xticks(rotation=45)
    save_plot(plt, 'weekday_status.png')

    # 14. Nationality vs Gender
    top_nationalities = df['Nationality'].value_counts().nlargest(5).index
    sns.countplot(data=df[df['Nationality'].isin(top_nationalities)], x='Nationality', hue='Gender')
    plt.title('Top Nationalities by Gender')
    save_plot(plt, 'nationality_gender.png')

    # 15. Flight Status Pie Chart
    status_counts = df['Flight Status'].value_counts()
    plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Flight Status Distribution')
    save_plot(plt, 'flight_status_pie.png')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            return redirect(url_for('dashboard'))
        else:
            return "Login Failed. Check your credentials."
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (fullname, email, username, password) VALUES (?, ?, ?, ?)', 
                       (fullname, email, username, password))
        conn.commit()
        conn.close()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(UPLOADS_PATH, filename)
            file.save(filepath)

            # Load dataset
            df = pd.read_csv(filepath)
            df.dropna(inplace=True)
            df['Departure Date'] = pd.to_datetime(df['Departure Date'], errors='coerce')
            df.dropna(subset=['Departure Date'], inplace=True)

            # Generate EDA visualizations
            generate_eda(df)

            images = os.listdir(PLOTS_PATH)
            flash('Dataset uploaded and visualizations generated!', 'success')
            return render_template('dashboard.html', images=images)

    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)
