import yfinance as yf
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

################################### GETTING DATA ###################################################

# Converts all of the dates into a proper date format
def convert_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(int(timestamp) / 1000).strftime('%Y-%m-%d')

# Gets the specified stock data 
stock_data = yf.download("AAPL", start="2024-01-01", end="2024-08-13")
stock_data = stock_data[['Close']] 
stock_data['Date'] = stock_data.index.date  # Convert index to date
stock_data.reset_index(drop=True, inplace=True)
stock_data = stock_data[['Date', 'Close']]

# Gets the API request of the headlines containing the stock name
url = "https://google-news13.p.rapidapi.com/search"
querystring = {"keyword":"facebook","lr":"en-US"}
headers = {
	"x-rapidapi-key": "f4c09b4829msh9d13b47c4dd1d90p19eb1fjsndfff5a9e6c8f",
	"x-rapidapi-host": "google-news13.p.rapidapi.com"
}
response = requests.get(url, headers=headers, params=querystring)
news_data = response.json()

# Creates a date source with title and timestamp of the headlines
headlines = [item['title'] for item in news_data['items']]
headline_dates = [convert_timestamp(item['timestamp']) for item in news_data['items']]

# Gets the sentiment fot the headlines 
analyzer = SentimentIntensityAnalyzer()
headline_sentiments = []
for item in news_data['items']:
    headline = item['title']
    sentiment = analyzer.polarity_scores(headline)['compound']
    date = convert_timestamp(item['timestamp'])
    headline_sentiments.append({'Date': date, 'Sentiment': sentiment})
sentiment_df = pd.DataFrame(headline_sentiments)

# Makes a dataframe of the daily sentiments
daily_sentiment = sentiment_df.groupby('Date').mean().reset_index()  

# Ensure the Date columns are both datetime in the two data frames
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])

# Merge stock data with sentiment data
merged_data = pd.merge(stock_data, daily_sentiment, on='Date', how='left')
merged_data['Sentiment'].fillna(0, inplace=True)  # Fill missing sentiment data with 0

# Create labels for price movement (1 for up, 0 for down)
merged_data['Price_Change'] = merged_data['Close'].diff().shift(-1)
merged_data['Up_Down'] = (merged_data['Price_Change'] > 0).astype(int)
# print(merged_data[['Close', 'Sentiment', 'Up_Down']])


################################### TRAINING THE MODEL ###################################################
# Features and labels
X = merged_data[['Sentiment']]
y = merged_data['Up_Down']

# Train-test split with 25% of the data being used to test the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print(y_pred)


################################### PLOT RESULTS ###################################################
# Align the predictions with the corresponding dates from the test set
test_dates = merged_data['Date'].iloc[-len(y_test):].reset_index(drop=True)

# Create the DataFrame for plotting
predictions_df = pd.DataFrame({
    'Date': test_dates,  # Use only the test set dates
    'Actual': y_test.reset_index(drop=True),
    'Predicted': y_pred
})

# Plot the actual vs. predicted values
plt.figure(figsize=(14, 7))
plt.plot(predictions_df['Date'], predictions_df['Actual'], label='Actual Price Movement', color='blue', marker='o')
plt.plot(predictions_df['Date'], predictions_df['Predicted'], label='Predicted Price Movement', color='red', linestyle='--', marker='x')

plt.title('Actual vs Predicted Stock Price Movement')
plt.xlabel('Date')
plt.ylabel('Movement (1 = Up, 0 = Down)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

print(plt.show())

