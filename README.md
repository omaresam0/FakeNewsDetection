# Fake News Detection
The project aims to build a model that can classify news articles as either real or fake based on their content.

## Dataset

News dataset contains labeled news articles. Each article is labeled as "REAL" or "FAKE" based on its authenticity.

## Algorithms
### Support Vector Machines (SVM): 
SVM is a powerful algorithm for text classification tasks. It effectively separates data points using hyperplanes and works well in high-dimensional spaces. SVM is capable of capturing complex relationships between textual features and class labels, making it suitable for fake news detection.

### Naive Bayes:
Naive Bayes is a probabilistic algorithm that applies Bayes' theorem with the assumption of independence between features. It works well with text classification tasks and is computationally efficient. Naive Bayes models the probability of a news article being fake or real based on the occurrence of words in the text.

### Gradient Boosting:
Gradient Boosting is an ensemble method that combines multiple weak learners (decision trees) to create a strong predictive model. It builds the model in an iterative manner, focusing on instances that are difficult to classify correctly. Gradient Boosting can capture complex relationships and interactions between features, leading to improved accuracy.

### Random Forest:
Random Forest is another ensemble learning algorithm that combines multiple decision trees. It operates by constructing a multitude of decision trees and averaging their predictions. Random Forest can handle high-dimensional data and tends to provide robust performance in classification problems

### Decision Trees:
Decision trees are intuitive and interpretable models that partition the feature space based on simple rules. They can capture non-linear relationships between features and labels. However, decision trees may not always provide a high accuracy.

## Preprocessing

1. Lowercasing: The text data is converted to lowercase to ensure consistency.

2. Punctuation Removal: Punctuation marks are removed from the text to eliminate unnecessary noise.

3. Tokenization: The text is tokenized into individual words or tokens to prepare it for further processing.

4. Stopword Removal: Stopwords, such as common words like "and" or "the," are removed from the text as they often don't carry significant meaning.

5. Lemmatization: Words are lemmatized to reduce them to their base or root form. This helps in reducing word variations and improving the consistency of the data.

These preprocessing steps help in cleaning the text data, reducing noise, and improving the quality of features for the machine learning model.

## Results

SVM achieved the highest accuracy among the algorithms tested, It achieved an accuracy of 0.94%, precision of 0.94%, recall of 0.93%, and F1 score of 0.94%.

Decision Trees yielded the lowest accuracy among the algorithms tested, with an accuracy of 0.80%, precision of 0.79%, recall of 0.80%, and F1 score of 0.80%.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request.
