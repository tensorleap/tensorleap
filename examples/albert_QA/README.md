# Albert model with SQuAD dataset

This project implements the Albert algorithm using the [SQuAD]() (Stanford Question Answering Dataset) for question
answering tasks.

## Population Exploration

Below is a population exploration plot. It represents a samples' similarity map based on the model's latent space,
built using the extracted features of the trained model.

It shows a visualization of the training and validation sets where we can see two distinct clusters. 
This means that there is a difference in their representation that might indicate some sort of imbalance.

![Latent space_dataset_state](screen_shots/population_exploration_dataset_state.png)

### *Detecting & Handling High Loss Clusters*
In Question Answering (QA), the "title" refers to the title of the passage from which the question is derived, one of
the titles in the dataset is ‘American Idol’.
Further analysis reveals that a cluster in samples related to ‘American Idol’ title, has a higher loss
(larger dot sizes).
At a glance, we can see that this cluster contains questions that relate to names of songs, 
such as “what was [singer’s name] first single?” or “what is the name of the song…?”.

![High Loss Clusters_American_idol](screen_shots/High_loss_clusters_american_idol.png)

It appears that the model did detect the correct answers. However, the prediction contains quotation marks while the
ground truth doesn’t.

### *Detecting Unlabeled Clusters in the Latent Space*

Now, let’s look for additional clusters in our data using an unsupervised clustering algorithm on the model’s latent
space.
Upon examination of these clusters, we can see that clusters 13 and 20, located close to each other, contain answers 
relating to years and dates of events. Cluster 20 (left side image) includes primarily questions that require answers
related to years, 
such as “What year…?” where the answers are years represented in digits: “1943”, “1659” etc. 
Cluster 13 (right side image), includes questions that require answers related to dates and times, such as “When.. ?” 
and answers of the dates and times represented in text and digits: “early months of 1754”, “1 January 1926”, 
“20 december 1914”, “1990s” etc.

<div style="display: flex">
  <img src="screen_shots/cluster_20.png" alt="Image 1" style="margin-right: 10px;">
  <img src="screen_shots/cluster_13.png" alt="Image 2" style="margin-left: 10px;">
</div>

### *Fetching similar samples*

Another approach to finding clusters using the model’s latent space is fetching similar samples to a selected sample.
It enables you to identify a cluster with an intrinsic property you want to investigate. 
By detecting this cluster, you can gain insights into how the model interprets this sample and, in general, retrieve 
clusters with more abstract patterns.

The figure below shows a Quantitative Questions and Answers cluster. We can see that the cluster which includes 
quantitative questions and answers contains questions such as “How many …?”, “How often…?”, “How much …?” and answers 
represented in digits and in words: “three”, “two”, “75%”, “50 million”, “many thousands”.

![fetching_similar_samples](screen_shots/fetching_similar_samples.png)

### *Sample Loss Analysis*
In this section, we can see the results of a gradient-based explanatory algorithm to interpret what drives the model to 
make specific predictions. It is enables us to analyze which of the informative features contributes most 
to the loss function. We then generate a heatmap with these features that shows the relevant information.

Let’s analyze the following sample containing the question: “when did Beyonce release ‘formation’?”. The correct 
predicted answer is: “February 6, 2016”. We see that the tokens that had the most impact on the model’s prediction are:
‘when’, ‘one’, ‘day’, ‘before’. Also, the answer tokens:’ february’, ‘6’,’ 2016′.

![Sample Loss Analysis](screen_shots/Sample_Loss_Analysis.png)

### *False / Ambiguous Labelings*

The figure below shows an example for illustrates inaccurate and mislabeled samples.
We can see a sample with the question (shown in purple): “Did Tesla graduate from the university?” The answer from the 
context is: “he never graduated from the university” (shown in orange). This was detected correctly by the model. 
However, the ground truth’s indexes refer to “no” (shown in green) in the sentence: “no Sundays or holidays…”. As in 
the above example, the indexes are incorrect and in an unrelated location to the question.

![False / Ambiguous Labelings](screen_shots/Ambiguous Labelings.png)

Such cases distort the model’s performance and negatively affect its fitting when penalizing the model on these samples.
We can solve these issues by changing the ground truth or by removing such samples.




