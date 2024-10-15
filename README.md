# [Bayer Hackathon 2024]](https://github.com/karenbolon/Bayer_2024)

Our team won the category 'Sensitivity and Uncertainty in Predictive Models' at the Bayer and Max Delbrück Center "Data Science Meets Biology Hackathon" Hackathon 2024.  It was a collaborative effort with 4 other people (2 from Max Delbrück Center and 2 from 42 Berlin).

## Project Outline
Our project focused on analyzing how point mutations in amino acid sequences affect nanobody thermostability. Leveraging our knowledge of biochemistry and amino acid properties, we quantified shifts in model predictions and evaluated their consistency with anticipated results. This interdisciplinary project—spanning bioinformatics, molecular biology, and machine learning—was an incredible experience, showcasing the transformative potential of computational tools in advancing pharmaceutical research.

Trained model used (TEMPRO model):  https://github.com/Jerome-Alvarez/TEMPRO/tree/main Goals:



____




# [Bayer Challenge 2023](https://github.com/karenbolon/Bayer_challenge)

As part of the Bayer Coding Challenge, this project was completed in collaboration with 3 other 42Berlin students.  This project visualizes the Wisconsin Breast Cancer dataset in an interactive dashboard which offers exploratory data analysis, machine learning model predictions, and AI-driven insights about the data. The goal is to find valuable patterns and train a machine learning model to improve breast cancer diagnosis.

Features:
Integrated OpenAI for answering dataset-related queries.
Includes SHAP analysis to interpret the models' predictions.
You can access the dashboard via the following:

Online Dashboard
Due to security, we could not keep a working version of our streamlit app.  Please watch our quick video showcasing our app: 

[![Watch the video](https://img.youtube.com/vi/gaWY_N0boyg/maxresdefault.jpg)](https://www.youtube.com/watch?v=gaWY_N0boyg)


____



# [Image Classification Using Neural Networks](https://github.com/kbolon1/Diabetic_Image_Classifier)

Trained a neural network classification model (ResNet152) with transfer learning to identify between a healthy eye or an eye suffering from diabetic retinopathy with over 10,000 images (5000 images of each class) using data augmentation.  This aims to improve the mass screening of populations and eventually decrease medical costs through computer-aided diagnosis. This is crucial as the number of cases and the prevalence of diabetes have steadily increased over the past few decades with approximately 422 million people diagnosed worldwide. Diabetic retinopathy is a complication of diabetes, caused by high blood sugar levels that damage the back of the eye.  It can cause blindness if left undiagnosed and untreated.  It is also the leading cause of vision impairment in the world and is 100% avoidable.

![]()
<p float="left" align="center">
<img src="./images/Diabetes_Plotly_2011.png" title="Prevalence of Diabetes in 2011 per the Worldbank" width="500" height="325">
<img src="./images/Diabetes_Plotly_2021.png" title="Prevalence of Diabetes in 2021 per the Worldbank"  width="500" height="325">
</p>

Used Plotly with data pulled from the World Bank (https://databank.worldbank.org/home.aspx)

I used the unprocessed images found on Kaggle:
 - https://www.kaggle.com/competitions/diabetic-retinopathy-detection
 - https://www.kaggle.com/competitions/aptos2019-blindness-detection
 - https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

**Method:**
* Created a Deep-Learning Environment for Tensorflow and Keras and transfered the work to Google Colab .
* Used Keras Preprocessing on the images to resize and to augment
* Trained it on a sample of 5000 images of each class (10000 in totals) with only 50% accuracy rate. 


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
