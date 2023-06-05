from datetime import datetime
import logging

from decouple import config
import matplotlib.pyplot as plt
from meteostat import Daily
from meteostat import Point
from meteostat import units
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
import tweepy

# init logging and API credentials
logger = logging.getLogger()

bearer_token = config("BEARER_TOKEN")
consumer_key = config("CONSUMER_KEY")
consumer_secret = config("CONSUMER_SECRET")
access_token = config("ACCESS_TOKEN")
access_token_secret = config("ACCESS_TOKEN_SECRET")

# attempt to authenticate API V2 credenitals
try:
    client = tweepy.Client(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
    )
    logger.info("Tweepy V2 API credentials successfully verified.")
except Exception:
    logger.info("Failed to authenticate Tweepy V2 API credentials.")

# attempt to authenticate API V1 credenitals
try:
    auth = tweepy.OAuth1UserHandler(
        consumer_key, consumer_secret, access_token, access_token_secret
    )
    api = tweepy.API(auth)
    logger.info("Tweepy V1 API credentials successfully verified.")
except Exception:
    logger.info("Failed to authenticate Tweepy V1 API credentials.")

# Fetch weather data from metostat API starting today and heading back N years
N = 75
end = datetime.now()
start = datetime(end.year - N, end.month, end.day)

# Define location of NYC
nyc = Point(40.7789, -73.9692, 3.0)

# Fetch the weather series from NYC
data = Daily(nyc, start, end)
data = data.convert(units.imperial)
data = data.fetch()

# Create indexer columns to make a day of year pivot table
data["year"], data["month"], data["day"] = (
    data.index.year,
    data.index.month,
    data.index.day,
)

# Create a pivot table of daily high and daily average temperature for today's date over last N years
daily_max_avg = pd.DataFrame(data.tmax.groupby(by=[data.month, data.day]).mean())
daily_max_max = pd.DataFrame(data.tmax.groupby(by=[data.month, data.day]).max())

# Create month, day tuple
daily_max_avg["time"] = daily_max_avg.index.values
today = end.strftime("%Y-%m-%d")

# find today's high
todays_high = data.tmax.loc[today]

# store todays date
Month, Day, Year = end.month, end.day, end.year

# store the avg temp and max temp for today
todays_avg_high = daily_max_avg.tmax.loc[(Month, Day)]
todays_max_high = daily_max_max.tmax.loc[(Month, Day)]

# generate dataframe for scatter plot
pd.options.mode.chained_assignment = None
daily_hist_series = data[(data["month"] == Month) & (data["day"] == Day)]
daily_hist_series["most_recent"] = np.where(
    daily_hist_series["year"] == daily_hist_series["year"].max(), 1, 0
)

# find the year of the max temp on today's date
todays_max_high_year = np.argmax(daily_hist_series[["tmax"]].values) + start.year

# Compare weather conditions of today to the average high
if todays_high > todays_avg_high:
    forecast_tweet = f"NYC: The high today is {todays_high:.1f}°F, which is {abs(todays_high - todays_avg_high):.1f}°F hotter than today's {N}-year average."
else:
    forecast_tweet = f"NYC: The high today is {todays_high:.1f}°F, which is {abs(todays_high - todays_avg_high):.1f}°F cooler than today's {N}-year average."

forecast_tweet += f"\n\nThe {N}-year historical high for today of {todays_max_high:.1f}°F  was set in {todays_max_high_year}."

# Fit a spline to create a trend line over the last N years
x, y = daily_hist_series["year"].values.reshape(-1, 1), daily_hist_series[
    ["tmax"]
].values.reshape(-1, 1)
model = make_pipeline(SplineTransformer(n_knots=4, degree=2), Ridge(alpha=1e-3))
model.fit(x, y)

# Spread out the y vector to correctly calculate the the jackknive SEs
y_true_long = np.linspace(y.min(), y.max(), 500).reshape(-1)
x_plot = np.linspace(Year - N, Year, 500).reshape(-1, 1)
y_plot = model.predict(x_plot).reshape(-1)

# Calculate jackknive SEs in dimensions suitable for plotting
residuals = (y_true_long - y_plot).reshape(-1)
res_sd = np.std(residuals)
leverage = np.diagonal(x_plot.dot(np.linalg.inv(x_plot.T.dot(x_plot)).dot(x_plot.T)))
jackknife_se = residuals / (res_sd * np.sqrt(1 - leverage))

upper_jk_ci = y_plot + (1.96 * jackknife_se)
lower_jk_ci = y_plot - (1.96 * jackknife_se)

# Create dicts to format months and days in the graph output
months = "January February March April May June July August September October November December".split(
    " "
)

months_formatter = {idx + 1: month for (idx, month) in enumerate(months)}
days_formatter = {
    1: "st",
    2: "nd",
    3: "rd",
    4: "th",
    5: "th",
    6: "th",
    7: "th",
    8: "th",
    9: "th",
    0: "th",
}

# plot the weather data
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (16, 9)

plt.scatter(
    daily_hist_series["year"][daily_hist_series["most_recent"] == 0],
    daily_hist_series["tmax"][daily_hist_series["most_recent"] == 0],
    color="gray",
    zorder=3,
)

plt.scatter(
    daily_hist_series["year"][daily_hist_series["most_recent"] == 1],
    daily_hist_series["tmax"][daily_hist_series["most_recent"] == 1],
    color="red",
    label="Today",
    zorder=4,
)

plt.plot(x_plot, y_plot, color="Black", zorder=2, label="Trend")

plt.fill_between(
    x_plot.reshape(-1),
    upper_jk_ci,
    lower_jk_ci,
    color="tomato",
    zorder=1,
    alpha=0.3,
    label="95% Confidence Interval of Trend",
)

plt.xlabel("Year")
plt.ylabel("Daily High in Degrees °F")
plt.title(
    f"Daily High on {months_formatter[Month]} {Day}{days_formatter[Day%10]} by Year"
)
plt.legend()
plt.savefig("daily_plot.jpeg")

# upload the media to the Tweepy API
media = api.media_upload(filename="daily_plot.jpeg")

# tweet the takeaway
response = client.create_tweet(text=forecast_tweet, media_ids=[media.media_id_string])
logger.info(
    f"Today's forecast tweeted: https://twitter.com/user/status/{response.data['id']}"
)
print(
    f"Today's forecast tweeted: https://twitter.com/user/status/{response.data['id']}"
)
