# Portfolio
My Portfolio

# [Project: Markov Chain Monte Carlo](https://github.com/spicedacademy/fenugreek-student-code/tree/karen/week_08_project)

* Created a simulation of customer behavior in a supermarket using Markov-Chain Monte Carlo methods.
* Used Pandas and NumPy for data wrangling, calculated a transition probabilities matrix, implemented a customer class, and then run a MCMC simulation for customer(s).

<img src="https://github.com/kbolon1/Portfolio/blob/main/images/MCMC_EDA.png" width="500" height="350">

# [Project: Neural Networks Image Classification](https://github.com/spicedacademy/fenugreek-student-code/tree/karen/week_09_project)

* Built an image classifier by collecting my own photos through a webcam to make predictions on images
* Created a deep learning environment to run tensorflow and keras
* Built a neural network from scratch and then used transfer learning with a pretrained model (MobileNetV2) with transfer learning

<img src="https://github.com/kbolon1/Portfolio/blob/main/images/5Things_accuracy.png" width="600" height="400">
<img src="https://github.com/kbolon1/Portfolio/blob/main/images/Predictions.png" width="400" height="300">
<img src="https://github.com/kbolon1/Portfolio/blob/main/images/Flower.png" width="200" height="200">

# [Project: Recommender Systems](https://github.com/spicedacademy/fenugreek-student-code/tree/karen/week_10_project)

* Built a web application that showcases movie recommendations using the small version of the MovieLens-dataset (https://grouplens.org/datasets/movielens/)
* Performed data engineering on missing data(KNN Imputer, Mean Movie Ratings).
* Implemented the following models trying to find the lowest Mean Error :
	- Simple recommender based on correlations
	- Non-Negative Matrix Factorization model
	- KMeans (clustering)
	- Nearest Neighbour
* Wrote a flask web interface and connected the recommender-model to flask

<img src="https://github.com/kbolon1/Portfolio/blob/main/images/Movie_Recommender_Main.png" width="600" height="400">
<img src="https://github.com/kbolon1/Portfolio/blob/main/images/Movie_Recommender_Second.png" width="600" height="400">

# [Final Project: Image Classification Using Transfer Learning on ResNet152 Model](https://github.com/spicedacademy/fenugreek-student-code/tree/karen/Project)

Goal: to trained a model to distinguish between an eye suffering from diabetic retinopathy (blindness caused by high blood sugar levels) and a healthy eye. 
![](https://github.com/kbolon1/Portfolio/blob/main/images/Eye_Fundus.png)
Datasets from Kaggle, I used the preprocessed images:
	- https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k
	- https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
	- https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data

* Used Plotly to show the prevalence of diabetes with data pulled from the World Bank (https://databank.worldbank.org/home.aspx)
![](https://github.com/kbolon1/Portfolio/blob/main/images/Diabetes_Plotly_2011.png)
![](https://github.com/kbolon1/Portfolio/blob/main/images/Diabetes_Plotly_2021.png)
* Created a Deep-Learning Environment for Tensorflow and Keras at the beginning and then moved to Google Colab as the sample size were large
* Used Keras Preprocessing on the images to resize and to augment
* Built my own CNN model with 3 Convoluted layers with MaxPooling, a Flattening layer, and 3 Dense Layers with Dropout and Batch Normalization while using a mix of RELU and Sigmoid activations and selected Categorical_crossentropy for loss function.
* Tried a VGG19 model and then a ResNet152 model with transfer learning. Eventually trained it on a sample of 5000 images of each class with only 50% accuracy rate. I am still working on improving this.

# [Project: GIFs of Fertility Rate vs Life Expectancy](https://github.com/spicedacademy/fenugreek-student-code/tree/karen/Gapminder_Analysis_GIF_Project)

* Collected data from gapminder project (https://www.gapminder.org/data/)
* Used pandas to import and merge data 
* Created GIF using Seaborn and ImageIO

	![](https://github.com/kbolon1/Portfolio/blob/main/images/gapminder_output.gif)

# [Project: Titanic Machine Learning from Disaster (Predicting Survival Rates)](https://github.com/spicedacademy/fenugreek-student-code/tree/karen/week_02_project)

* Collected data from Kaggle (https://www.kaggle.com/c/titanic)
* Built and compared a variety of Machine Learning classifiers with scikit-learn (logistic regression, support vector machine, random forest) to predict survival of passengers on the Titanic
* Project consisted of all phases of Machine Learning work-flow (train-test-splitting of data, data exploration, feature engineering, optimization of hyperparameters, evaluation of models with cross-validation)

A graph showing those that survived vs perished by age.
	![](https://github.com/kbolon1/Portfolio/blob/main/images/titanic_graph.png)
 
A confusion matrix on the Random Forest Model: 
 	![](https://github.com/kbolon1/Portfolio/blob/main/images/titanic_confusionmatrix.png)
	
# [Project: Capital Bike Sharing](https://github.com/spicedacademy/fenugreek-student-code/tree/karen/week_03_project)

* Built and trained a regression model on the Capital Bike Share (Washington, D.C.) Kaggle data set (https://www.kaggle.com/competitions/bike-sharing-demand/data), in order to predict demand for bicycle rentals at any given hour, based on time and weather
* Imported and cleaned data, performed exploratory data analysis (EDA) using Pandas 

	![](https://github.com/kbolon1/Portfolio/blob/main/images/bike_rentals.png)
	![](https://github.com/kbolon1/Portfolio/blob/main/images/bike_graphs6.png)
	![](https://github.com/kbolon1/Portfolio/blob/main/images/bike_heatmap.png)
	
* Performed Data Engineering using Pipelines, ColumnTransformer, OneHotEncoder, MinMaxScaler, StandardScaler, and RobustScaler
* Trained regression models (Random Forest Regression, Linear Regression, Polynomial Regression with Cross Validation) to find the best Root Mean Squared Log Error (RMSLE)  
* Used ElasticNet to regularise the model
* Used Hyperparameter Optimization on Random Forest Regression Model and GridSearchCV

<img src="https://github.com/kbolon1/Portfolio/blob/main/images/bike_result_RMSLE.png" width="450" height="200">

# [Project: Text Classification with Webscraping](https://github.com/spicedacademy/fenugreek-student-code/tree/karen/week_04_project)
* Built a text classification model on song lyrics, the following was performed: 
	- Downloaded an HTML page with links to songs (https://www.lyrics.com)
	- Extracted hyperlinks of songs to download and extract song lyrics
	- Vectorized the text using the Bag Of Words method and normalised the word counts with term frequency-inverse document frequency (TF-IDF)
	- Trained a classification model (logistic regression model) that predicts the artist from a piece of text

<img src="https://github.com/kbolon1/Portfolio/blob/main/images/maplehip.png" width="250" height="250">    
<img src="https://github.com/kbolon1/Portfolio/blob/main/images/stargwen.png" width="250" height="250">       

# [Project: Dashboard using Northwind Database](https://github.com/spicedacademy/fenugreek-student-code/tree/karen/week_05_project/northwind)

* Build a dashboard summarizing the Northwind Database (a sample database from Microsoft Access). 
* Used PostgreSQL to create tables in database, uploaded the data AWS EC2 Ubuntu server and then connected to Metabase to create a Dashboard.

# [Project: Twitter Sentiment Project](https://github.com/spicedacademy/fenugreek-student-code/tree/karen/week_06_project)

* Built a data pipeline with Docker-compose that collected tweets and stored them in a MongoDB database. 
* Created an ETL job that pulled the tweets from MongoDB for sentiment analysis and then stored the analysed tweets on a second database (PostgreSQL).

<img src="https://github.com/kbolon1/Portfolio/blob/main/images/Twitter_Sentiment_George_Takei.png" width="500" height="300"> 

# [Project: Time Series Analysis](https://github.com/spicedacademy/fenugreek-student-code/tree/karen/week_07_project)

* Created a short-term temperature forecast using data from (www.ecad.eu).
* Built a baseline model modelling trend and seasonality, plotted and inspected the different components of a time series.
* Used the following models to model the time dependence of the remainder:
	- Linear Regression
	- Autoregression
	- Auto Regressive Integreated Moving Average (ARIMA)
* Evaluated the model using Cross-Validation Time Series Split

<img src="https://github.com/kbolon1/Portfolio/blob/main/images/Temp_Seasonality.png" width="500" height="300"> 
<img src="https://github.com/kbolon1/Portfolio/blob/main/images/ARIMA_chart.png" width="500" height="300"> 


