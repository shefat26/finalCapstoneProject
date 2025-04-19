# Multiclass Movie Genre Classificiation

This project leverages a sequential Keras model to categorize genres based on the movie plot description into 17 distinct genres. The model employs GloVe word embeddings and a Bidirectional LSTM architecture for accurate genre classification. The goals is to develop a genre classifying model to precisely categorize the genres based on the movie/book plot description.

## Dataset

The model was trained on a dataset from the ["Multi Label Film Classifier"](https://www.kaggle.com/datasets/mdzarinhossain/multi-label-film-classifier), which classifies genres into seventeen categories. those are, 

| Genre       | Genre       | Genre       | Genre       | Genre      |
|-------------|-------------|-------------|-------------|------------|
| Action      | Adventure   | Animation   | Biography   | Comedy     |
| Crime       | Documentary | Drama       | Family      | Fantasy    |
| Film-Noir   | Game-Show   | Horror      | Mystery     | Reality-TV |
| Romance     | Sci-Fi      | Sport       | Thriller    |            |


## Dataset Format (Data Example):

| Title                         | Category | Metascore | User Score | Genres                                    |
|------------------------------|----------|-----------|------------|--------------------------------------------|
| Dekalog (1988)               | movie    | 100       | 100        | ['Drama']                                  |
| The Godfather                | movie    | 100       | 100        | ['Crime', 'Drama']                         |
| Lawrence of Arabia (re-release) | movie | 100       | 100        | ['Adventure', 'Biography', 'Drama', 'War'] |
| The Leopard (re-release)     | movie    | 100       | 100        | ['Drama', 'History']                       |
| The Conformist               | movie    | 100       | 100        | ['Drama']                                  |



### The dataset is stored in a CSV format with two important columns:
- Plot_summary: The description of a movie plot. (Feature Column)
- Genres: The genre class label/labels associated with each Plot summary. (Target Column)
- Other columns are irrelevant, and hence are dropped.

## Preprocessing

Plot summaries are cleaned to eliminate irrelevant information and reduce noise and nuances. Such as,
- Text is lowercase.
- Contractions are expanded.
- URLs, mentions, and special characters are removed.
- Multi-genre rows were converted to single genres.

## Exploratory data analysis

![Word-Character Distribution](https://raw.githubusercontent.com/RezuwanHassan262/sifat_repo/refs/heads/main/figures/wcc_dist.png) 

- There are two main plot length styles in the dataset — brief (30–40 words) and extended (60–70 words).
- A maximum character limit of ~342 is enforced or commonly hit.
- The dataset mixes single-sentence summaries with multi-sentence overviews.
- Useful for training models where input size and richness vary, such as in summarization or classification tasks

![Genre Distribution](https://raw.githubusercontent.com/RezuwanHassan262/sifat_repo/refs/heads/main/figures/genre_count.png) 
- There are 27 different unique genres but later Genre counts that had less than 10 were dropped.
- Drama Genre is most often
- There are two main plot length styles in the dataset — brief (30–40 words) and extended (60–70 words).

![Label cardinality](https://raw.githubusercontent.com/RezuwanHassan262/sifat_repo/refs/heads/main/figures/label_cardi.png) 
- Label cardinality: Most of the movies have 3 genres

![Genre co-occurrence](https://raw.githubusercontent.com/RezuwanHassan262/sifat_repo/refs/heads/main/figures/co_occurance.png) 
- Genre co-occurrence: Drama is being classified as Drama very often


![Word cloud](https://raw.githubusercontent.com/RezuwanHassan262/sifat_repo/refs/heads/main/figures/word_cloud.png) 
- Word cloud indicates life, new, young, world, family, man, love focuses on personal journeys, relationships, and family dynamics and frequent themes of youth and self-discovery

![Word cloud](https://raw.githubusercontent.com/RezuwanHassan262/sifat_repo/refs/heads/main/figures/tf_idf_table.PNG) 
- Top phrases: year old, new york, high school, los angeles emphasises on age/life stage, urban settings, and school themes
- Common trigrams are new york city, world war ii, based true story and plots include historical events, real locations, and biographical elements which mentions of production companies suggest some summaries mix marketing content
- High-frequency verbs/nouns: find, discover, follow, story, family indicates themes of transformation, search, and human-centered stories
- Strong presence of drama, family, coming-of-age, romance, action settings often in big cities or historic periods


The data was cleaned, normalized, and prepared for modeling.


## Modeling

- Logistic Regression 
- Random Forest
- Custom Sequential Keras Model (NLP Approach)

To train the machine learning algorithms such as Logistic Regression and Random Forest, _important_words_ were selected to help the models learn the plot summary better

For the Custom sequential modeling approach, followed these NLP practice as pre-processing steps,

- Tokenization is applied to convert tweets into sequences of words.
- Padding ensures all sequences are of equal length.
- Word Embeddings: GloVe embeddings ([glove.6B.100d](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt)) are used to map words to vectors.
 

## Model Architecture

The model is consisting of multiple layers including:

- Embedding Layer: Leveraged GloVe word embeddings to give meaning to words in the model.
    Bidirectional LSTM Layers: Employs a bidirectional approach to capture long-range dependencies in both forward and backward directions.
    Dropout and Batch Normalization: Regularization techniques to reduce model complexity and improve generalization.
    Fully Connected Layers: To integrate the extracted insights into a unified representation for decision-making.

A diagram of developed the model is given below

![Developed TF model](https://raw.githubusercontent.com/RezuwanHassan262/sifat_repo/refs/heads/main/figures/model_arch.png) 



 
## Results

After a lot of trials and errors with the parameters and model architecture, this model was finalized.

### Model Performance

We can observe the model's performance on training and validation data, the learning process, and its generalization ability.

![Model train-Loss Curves](https://raw.githubusercontent.com/RezuwanHassan262/sifat_repo/refs/heads/main/figures/train_valid_curves.PNG) 

It wasn't a good training performance because, 

1. Didn't have enough time to experiment with layers
2. Model is too simplistic to classify this much categories
3. Model architecture not too complex to understand/learn plot_summary text data properly
4. The dataset is too small and imbalanced

Model performance on test data.

| Model                         |  Accuracy |
|-------------------------------|-----------|
| Logistic Regression           |   ~49%    |
| Random Forest                 |   ~45%    |
| Custom Keras model            |   ~5%     |


### Evaluation Metrics

- Confusion Matrix: Confusion matrices is employed to assess the model's performance by visualizing the distribution of true positive, true negative, false positive, and false negative predictions for each sentiment category by all the models.

![Confusion Matrices](https://raw.githubusercontent.com/RezuwanHassan262/sifat_repo/refs/heads/main/figures/cfs.png)

#### From left: Logistic Regression - Random Forest - Keras Model

- Classification Report: A classification report analyzes the model's precision, recall, and F1-score for each sentiment category.

## Model Performance Metrics

![Classification Report](https://raw.githubusercontent.com/RezuwanHassan262/sifat_repo/refs/heads/main/figures/crcm.PNG) 



## Improvements

### Future work scopes:

    - Implementing transformer-based models like BERT to enhance sentiment classification precision.
    - Data augmentation methods are employed to expand tweet datasets, including back-translation and synonym replacement.
    - Hyperparameter optimization is conducted to enhance model performance.


<!-- 
## Explainable AI Integration (LIME: Local Interpretable Model-agnostic Explanations)

I aimed to enhance the interpretability of the model's predictions by integrating eXplainable AI (XAI) techniques. Specifically, I focused on implementing LIME (Local Interpretable Model-Agnostic Explanations). LIME works by approximating the complex model's behavior locally around a specific instance, creating a simpler, interpretable model to explain the prediction.

By incorporating LIME, I sought to understand the factors influencing the model's decisions. This would improve the model's transparency and help identify potential biases or errors.****
-->

