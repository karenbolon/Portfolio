# [Image Classification Using Neural Networks](https://github.com/kbolon1/Diabetic_Retinopathy_Image_Classifier)

**Final project for Spiced Academy** 

Trained a neural network classification model (ResNet152) with transfer learning to identify between a healthy eye or an eye suffering from diabetic retinopathy with over 10,000 images (5000 images of each class) using data augmentation.  This aims to improve the mass screening of populations and eventually decrease medical costs through computer-aided diagnosis. This is crucial as the number of cases and the prevalence of diabetes have steadily increased over the past few decades with approximately 422 million people diagnosed worldwide. Diabetic retinopathy is a complication of diabetes, caused by high blood sugar levels that damage the back of the eye.  It can cause blindness if left undiagnosed and untreated.  It is also the leading cause of vision impairment in the world and is 100% avoidable.

![]()
<p float="left" align="center">
<img src="./images/Diabetes_Plotly_2011.png" title="Prevalence of Diabetes in 2011 per the Worldbank" width="400" height="325">
<img src="./images/Diabetes_Plotly_2021.png" title="Prevalence of Diabetes in 2021 per the Worldbank"  width="400" height="325">
</p>

Used Plotly with data pulled from the World Bank (https://databank.worldbank.org/home.aspx)

![]()
<img src="./images/Eye_Fundus.png" width="400" height="200">

I used the unprocessed images found on Kaggle:
 - https://www.kaggle.com/competitions/diabetic-retinopathy-detection
 - https://www.kaggle.com/competitions/aptos2019-blindness-detection
 - https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

**Method:**
* Created a Deep-Learning Environment for Tensorflow and Keras and transfered the work to Google Colab .
* Used Keras Preprocessing on the images to resize and to augment
* Trained it on a sample of 5000 images of each class (10000 in totals) with only 50% accuracy rate. 

**Currently:**
I have started over and changed the model to use the LazyAdam optimizer and will process the images differently.  I am researching best ways to preprocess the images.  Stay tuned for updates.

____

# [Markov Chain Monte Carlo: Predicting Customer Behavior](https://github.com/kbolon1/MCMC_Supermarket)

* This helps companies analyse typical clientele behavior in order to improve profitability and customer satisfaction.  MCMC can help companies understand their clients buying process to find the key levers where they can influence a buyer's decision(s). This can describe for example:  
    - How often do they visit this part of the website/store? and for how long?
    - How much will they likely spend? etc.
* Created a simulation of customer behavior in a supermarket using Markov-Chain Monte Carlo methods.
* Used Pandas and NumPy for data wrangling, calculated a transition probabilities matrix, implemented a customer class, and then run a MCMC simulation for customer(s).

![]()
<p align="center">
<img src="./images/MCMC_EDA.png" width="600" height="350">
</p>

____

# [Recommender Systems: Movie Recommender](https://github.com/kbolon1/Movie_Recommender)

* This helps entities estimate the products (in this case movies) their clients will most likely want to view.  This reduces costs by only providing products that are needed and improve client retention by improving client experience through well-suited movie recommendations.
* Built a web application that showcases movie recommendations using the small version of the MovieLens-dataset (https://grouplens.org/datasets/movielens/)
* Performed data engineering on missing data(KNN Imputer, Mean Movie Ratings).
* Implemented the following models trying to find the lowest Mean Error :
    - Simple recommender based on correlations
    - Non-Negative Matrix Factorization model
    - KMeans (clustering)
    - Nearest Neighbour
* Wrote a flask web interface and connected the recommender-model to flask

![]()
<p align="center">
<img src="./images/Movie_Recommender_Main.png" width="600" height="350">
<img src="./images/Movie_Recommender_Second.png" width="600" height="350">
</p>
	
____

# [Text Classification Project: Webscraping Lyrics](https://github.com/kbolon1/Web_Scraper)

* Built a text classification model on song lyrics, the following was performed: 
    - Downloaded an HTML page with links to songs (https://www.lyrics.com)
    - Parsed HTML for hyperlinks to extract and download song lyrics
    - Vectorized the text using the Bag Of Words method and normalised the word counts with term frequency-inverse document frequency (TF-IDF)
    - Trained a classification model (Logistic Regression, Naive Bayes/MultinomialNB) that predicts the artist from a piece of text
    - Created a shaped WordCloud for each artist

* Used Python, BeautifulSoup, RegEx, Glob, Pillow, WordCloud, Seaborn, NumPy, Pandas, SciKit-Learn

![]()
<p float="left" align="center">
<img src="./images/maplehip.png" width="400" height="400"/>    
<img src="./images/stargwen.png" width="400" height="400"/>  
</p>

____

# [Sentiment Analysis Project: Twitter API (Docker and MongoDB](https://github.com/kbolon1/Twitter_Pipeline)

* Spiced Academy Project

![]()
<p align="center">
<img src="./images/Twitter_workflow.png" width="650" height="350"> 
</p>

* This can help entities analyse target customer sentiment.  They can learn how their products or even themselves are viewed and adjust accordingly.
* Used Python, MongoDB, Tweeter APIs, tweepy, Docker.
* Built a data pipeline with Docker-compose that collected tweets and stored them in a MongoDB database. 
* Created an ETL job that pulled the tweets from MongoDB, cleans data and calculates compound sentiment score (Vader Sentiment Analysis) for sentiment analysis and then stored the analysed tweets on a second database (PostgreSQL).

![]()
<p align="center">
<img src="./images/Twitter_Sentiment_George_Takei.png" width="450" height="300" align="center"> 
</p>

____

# [Project: Short-Term Weather Forecast-Time Series Analysis](https://github.com/kbolon1/Time_Series_Analysis)

* This model analysed a large set of historical data, identified patterns in order to predict future outcomes. 
* Created a short-term temperature forecast using data from (www.ecad.eu).
* Built a baseline model modelling trend and seasonality, plotted and inspected the different components of a time series.
* Used the following models to model the time dependence of the remainder:
    - Linear Regression
    - Autoregression
    - Auto Regressive Integreated Moving Average (ARIMA)
* Evaluated the model using Cross-Validation Time Series Split

![]()
<p float="left" align="center">
<img src="./images/Temp_Seasonality.png" width="500" height="300" align="center"/> 
<img src="./images/ARIMA_chart.png" width="500" height="300" align="center"/> 
</p>

____

# [Visual Data Analysis: Gapminder Data with GIFs of Fertility Rate vs Life Expectancy](https://github.com/kbolon1/Gapminder)

* Visualised a large set of data for communication in an easily digestible format.
* Collected data from gapminder project (https://www.gapminder.org/data/)
* Used python, pandas, seaborn, matplotlib, and NumPy to wrangle and explore data
* Created GIF using Seaborn and ImageIO showing fertility rate vs life expectancy between 1960 - 2016

![]()
<p align="center">
<img src="./images/gapminder_output.gif" width="500" height="400">
</p>

____

# [Classification Project: Titanic Dataset (Predicting Survival Rates)](https://github.com/kbolon1/Titanic_Dataset)

* Analysed known data to find key predictive variables (or feature optimization) to enhance models for future data analysis. This model used historical data to predict survival rate on the Titanic.
* Used python, pandas, seaborn, NymPy, matplotlib, sklearn
* Collected data from Kaggle (https://www.kaggle.com/c/titanic)
* Implemented and compared a variety of Machine Learning classifiers with scikit-learn (logistic regression, support vector machine, random forest) to predict survival of passengers on the Titanic

![]()
<p align="center">
<img src="./images/Titanic_CrossAccuracyLogs.png" width="300" height="175">
</p>

* Project consisted of all phases of Machine Learning work-flow: 
    - train-test-splitting of data, 
    - data exploration, 
    - feature engineering, 
    - optimization of hyperparameters, 
    - evaluation of models with cross-validation.

![]()
<p align="center">
<img src="./images/titanic_graph.png" width="300" height="250">
</p>
	
____

# [Regression Analysis Project: Capital Bike Sharing](https://github.com/kbolon1/Regression_Analysis)

* This model retains clients by providing a seemless customer experience while using a fit-for-purpose inventory. This helps entities forecast customer demand and ultimately optimise their inventory and minimise capital expenditures.  
* Used Python, Pandas, NumPy, Matplotlib, Seaborn, math, SciKit-Learn 
* Built and trained a regression model on the Capital Bike Share (Washington, D.C.) from the Kaggle data set (https://www.kaggle.com/competitions/bike-sharing-demand/data) to predict demand for bicycle rentals at any given hour based on time and weather
* Imported and cleaned data, performed exploratory data analysis (EDA) 

	![](/images/bike_graphs6.png)
	
* Performed Data Engineering using Pipelines, ColumnTransformer, OneHotEncoder, MinMaxScaler, StandardScaler, and RobustScaler
* Trained regression models (Random Forest Regression, Linear Regression, Polynomial Regression, Cross Validation) to find the best Root Mean Squared Log Error (RMSLE)  
* Used Hyperparameter Optimization on Random Forest Regression Model and GridSearchCV then ElasticNet to regularise the model to prevent overfitting

![]()
<p align="center">
<img src="./images/bike_result_RMSLE.png" width="350" height="175">
</p>
	
____

# [Image Classification: Neural Networks Using Webcam](https://github.com/kbolon1/Image_Classification_Webcam)

* Built an image classifier by collecting photos through a webcam to make predictions on images
* Used Python, NumPy, Pillow, Seaborn, CV2, Glob, SciKit-Learn, TensorFlow, and Keras
* Created a deep learning environment to run tensorflow and keras
* Built a neural network from scratch and then used transfer learning with a pretrained model (MobileNetV2) with transfer learning

![]()
<p align="center">
<img src="./images/5Things_accuracy.png" width="700" height="300">
</p>
<p float="left" align="center">
<img src="./images/Predictions.png" width="400" height="300" align="center"/>
<img src="./images/Flower.png" width="200" height="200" align="center"/>
</p>

____

# [Project: Dashboard using Northwind Database (AWS/EC2/Ubuntu/Metabase)](https://github.com/kbolon1/Dashboard_PostgreSQL)

* Built a dashboard summarizing the Northwind Database (a sample database from Microsoft Access). 
* Used PostgreSQL to create tables in database (data modeling), uploaded the data AWS EC2 Ubuntu server and then connected to Metabase to create a Dashboard.
