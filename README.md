# URL Classification on Extracted Features using Deep Learning

# Abstract. 
The widespread adoption of the World Wide Web (WWW) has brought about a monumental transition toward e-commerce, online banking, and social media. This popularity has presented attackers with newer opportunities to scam the unsuspecting - malicious URLs are among the most common forms of attack. These URLs host unsolicited content and perpetrate cybercrimes. Hence classifying a malicious URL from a benign URL is crucial to enable a secure browsing experience. Blacklists have traditionally been used to classify URLs, however, blacklists are not exhaustive and do not perform well against unknown URLs. This necessitates the use of Machine Learning/Deep Learning as they im-prove the generality of the solution. In this paper, we employ a novel feature ex-traction algorithm using ‘urllib.parse’, ‘tld’ and ‘re’ libraries to extract static and dynamic lexical features from the URL text. IPv4 and IPv6 address groups and the use of shortening services are detected and used as features. Static features like https/http protocols used show a high correlation with the target variable. Various machine learning and deep learning algorithms were implemented and evaluated for the binary classification of URLs. Experimentation and evaluation were based on 450,176 unique URLs where MLP and Conv1D gave the best overall results with 99.73% and 99.72% accuracies and F1 Scores of 0.9981 and 0.9983 respectively.
Keywords: Malicious URL Detection, 1-Dimensional Convolutional Neural Network (Conv1D), Deep Learning, Machine Learning, Multi-Layer Perceptron (MLP)


# 1.	Introduction
Resources on the Internet are located using the Uniform Resource Locator (URL). A malicious URL is a link devised to facilitate scam attacks and frauds. Clicking on such URLs can download malware that compromises a machine's security and the data stored within it or expose users to phishing attacks. With the rapid expansion of the Internet, there has been a rise in such URL-based cyber-attacks making malicious URL attacks the most popular form [1] of cyber-attack. There are two main compo-nents of an URL [2]: The protocol being used is denoted by the protocol identifier; The IP address or a domain name to locate resources is specified by the resource name. It implies the existence of a structure within URLs that attackers exploit to deceive users into executing malicious codes, redirecting users to unwanted malicious websites or downloading malware. While the following attack techniques use mali-cious URLs to attack their targets [3-5]: Phishing, Drive-by Download and Spam. Phishing attacks refer to types of attacks that trick users into revealing sensitive in-formation by posing as a genuine URL. Drive-by-Download attacks take place when malware is unintentionally downloaded on a system. Finally, spams are unsought for messages to promote or phishing.
The increase in malicious URL [1] attacks in recent years have necessitated re-search in the area to help detect and prevent malicious URLs. Blacklists and rule-based detection are the basis of most URL detection techniques [6]. While this form of detection achieves high accuracy, with the variety of attack types and new ways to attack users and the vast magnitude of scenarios in which such attacks can take it is hard to build a general solution and that’s why blacklists do not perform well against newer URLs. These limitations have led researchers to employ Machine learn-ing techniques to classify a URL as malicious or benign [7 - 10] in search of a robust general solution. Machine Learning and Deep Learning have contributed a lot in var-ious fields like the diagnosis of tumours [11], diabetes classification [12], retinal dis-ease detection [13] and support for digital agricultural systems [14], algorithms like Naïve Bayes and Support Vector Machine have been used to study twitter senti-ments [15] and inferring chronic kidney failure [16]. In a Machine Learning ap-proach, URLs are collected, out of which, features are extracted that can be inter-preted mathematically by the model and learn a prediction function that can classify an unknown URL as malicious or benign.
In this paper, we use lexical features extracted from the URL and use Machine Learning algorithms like Logistic Regression, Random forest, K-Nearest Neighbor (KNN), XG-Boost, Gaussian NB, Decision tree, Multi-Layer Perceptron(MLP) and a Deep Learning technique like 1-dimensional convolutional neural network(Conv1D). The structure of this paper is as follows. Section 2 reviews some recent works in the literature on malicious URL detection. Data preparation in Section 3. Section 4 de-scribes the algorithms used. Experimental results are provided in Section 5. The paper concludes in Section 6.


## 2.	Related Work
This section summarizes some of the recent work in the field of malicious URL classi-fication.

2.1	Malicious URL classification based on signature
Researchers have applied signature-based URL classification for a long time. Here a blacklist of known Malicious URLs is established. Sun et al. [17] 2016 built Auto-BLG. It automatically generates blacklists. Adding prefilters ensures that the blacklist is always valid. When an URL is accessed, a database query is generated against the blacklist. If the URL exists in the database, a warning is generated. Although this way of classification works well for known URLs, it quickly fails to classify unknown URLs, and it is tedious to update a blacklist.
Prakash et al. in 2006 [6] proposed a predictive blacklisting technique called PhishNet. It is a predictive blacklisting technique that detects phishing attacks. It exploits the fact the attacks change the structure of the URL when launching a phish-ing attack. PhishNet collected 1.55 million children and 6000 parent URL in 23 days. It was concluded that similar phishing URLs were 90% similar to each other. PhishNet is 80 times faster than Google’s API on average. Like any blacklisting approach, this approach also suffers from the fact that blacklisting techniques simply do not per-form well against unknown URLs that haven’t previously been blacklisted.
A preliminary evaluation was presented in 2008 by Sinha et al. [18], where they evaluate popular blacklists with more than 7000 hosts from an academic site. It was concluded that these backlists have significant false negatives and false positives.

2.2	Machine Learning based URL Classification
URL datasets have trained on all three types of machine learning methods to achieve high accuracy even with unknown URLs. Supervised, Unsupervised and Semi-Supervised methods have been used to classify URLs.
Ma et al. [19] in 2009 used statistical attributes from URLs and compared the proposed classification model with Gaussian NB, Logistic Regression models, and Support Vector Machine experiments. This approach gained an accuracy of 97%. Although high accuracies were achieved, this approach is not scalable.
Sahoo et al. in their study [2] in 2017 evaluated many models such as a URL da-taset Naïve Bayes, Decision Trees, SVM, Logistic Regression, Random Forest, Online Learning, etc. It was studied that heavily imbalanced data can produce high results even if labels every URL is benign. In our paper, we properly balance the data to prevent false high accuracies.  While Vundavalli et al. [20] in 2020 used Logistic Re-gression, Neural Network and Naïve Bayes on parameters like URL and domain identity to detect phishing websites in their paper. Naïve Bayes got the highest accu-racy of 91%.
Aydin et al. [21] in 2020 presented methods for phishing URL detection. They ex-tracted 133 separate features and then used Gain Ratio and ReliefF algorithms for feature selection. Naïve Bayes, Sequential Minimal Optimization and J48 classifiers were used. They obtained modest results of 97.18% on J48 (Gain Ratio) and 98.47% on J48 (ReliefF). Bharadwaj et al. [22] in 2022 used GloVe to represent words as global vectors by considering global word-word co-occurrence statistics. They used SVM and MLP with statistical features and GloVe and obtained the highest accuracy of 89% with MLP. 
In our paper, we extract lexical features including Character, Semantic groups and Host-based groups. In Section 5 the accuracy of using said features will be given. 



## 3.	Data Preparation
This section gives a detailed explanation of the dataset used. The feature extraction methods. The pre-processing steps that were taken to make the dataset training ready.

3.1	Dataset Used
The dataset used was published on Kaggle [23] in 2019 by Siddharth Kumar. The acquired dataset was collected from various sources such as PhisTank a database of phishing URLs and Malware Domains Blacklist which is a database of URLs that infect the system with malware. The dataset contains 450,176 unique URLs. 77% of said URLs are benign and 23% malicious. The Dataset has 3 column attributes: 'in-dex', 'url', 'label', 'result'.

3.2	Feature Extraction
This dataset does not include any trainable features. Hence, we use the novel feature extraction algorithm to extract features from the given URLs in the dataset. The algo-rithm uses lamdas and iterates through all the URLs to extract the following features were extracted from the URL. Lexical features used: 
Hostname Length: uses “urlparse” from the urllib.parse library to extract the host-name string from the URL using the “urlparse(i).netloc” where “i” is the current URL in the iteration.  The string is then passed to the len() function to get the length of the string.
Url Length: calculated by passing the URL string to len().
Path Length: Output of “urlparse(i).path” is passed to the len() function to calculate the length of the path of the URL.
First Directory Length: Output of “urlparse(i).path” is split using “urlpath.split(‘/’)[1]”, whose length is calculated using len().
 '%' Count: “i.count(‘%’)” where “i” is the URL string.
Top-Level Domain Length: Its length is calculated by counting the number of char-acters in the Top-Level Domain obtained the “get_tld” function from the “tld” li-brary.
'@' Count: “i.count(‘@’)” where “i” is the URL string. 
'-' Count: “i.count(‘-’)” where “i” is the URL string.
'?' Count: “i.count(‘?’)” where “i” is the URL string.
'.' Count: “i.count(‘.’)” where “i” is the URL string.
'=' Count: “i.count(‘=’)” where “i” is the URL string.
Number of Directories: Count the number of “/” in the path of URL string.
'http' Count: “i.count(‘http’)” where “i” is the URL string. 
'www' Count: “i.count(‘www’)” where “i” is the URL string 
Digits Count: Count of numeric characters from the URL string. 
Letters Count: Count of alphabetical characters from the URL string.
 Binary Features: Use of Shortening URL: Searches from a dictionary for shortening services using “search(dict, i)” from “re” library where ‘i’ is the URL string and “dict” is the dictionary.
Use of IP or not: Searches from a dictionary of IP groups (IPv4 and IPv6) for IP groups using “search(dict, i)” from “re” library where ‘i’ is the URL string and “dict” is the dictionary.

3.3	 Data Exploration
An essential pre-processing step. In this sub-section, we explore the dataset and per-form feature selection. There were no missing values found on this dataset and there were no features that had zero variance. The feature called 'count-letters' had a high correlation of 0.97 with 'url-length' as seen in Fig. 2. It is higher than the threshold of 0.85. Hence we dropped the 'count-letters' column.

3.4	Over-Sampling
After splitting the data for training and testing, we oversample the training set to bal-ance the unbalanced data to eliminate any bias towards benign URLs which would hamper learning. The ‘malicious’ URL class was randomly oversampled to match 75% of benign URLs in terms of quantity. SMOTE technique was not used in this study due to its generating random values nature based on the k values. This is undesirable because it may produce undesirable noise in the training data that may not be reflective of the real-world URLs. Fig. 3. illustrates the workflow used in this study.
 

3.5	Train-Test Split and Hardware Used
The dataset was split in an 80:20 ratio with a random state of 1. 80% of the data was used for training the models and 20% for testing. All the models were run on Jupyter Notebook environment with Python 3, Scikit-Learn and Tensorflow 2.8 on a system running on Windows 11 with Intel i7 10th Generation and 16GB RAM at 3200MHz.


# 4.	Technology Used
This paper employs many machine learning algorithms such as Decision tree, Logistic Regression, SVM, KNN, XG-Boost, Random forest, Gaussian NB Multi-Layer Percep-tron (MLP) and a Deep Learning technique like 1-dimensional convolutional neural network (Conv1D). These techniques have been used to predict liver disease [24], medical disease analysis [25] and for the early prediction of heart disease [26]. Table 2 briefly describes all the models used in this paper.
Table 2. Tabular Illustration of Classifier Models used and their training parameters
Classifier 	Description
Logistic Regression (LR)	Utilizes a logistic function used for classifying categorical varia-bles. They are statistical models. The sigmoid function imple-mented in linear regression is used in logistic regression. {Pa-rameters: default}.
	
XGBoost	Boosting technique based on random decision tree ensemble algorithm. Residual error updates after every tree learns from the preceding tree. Can handle sparse and also can implement regu-larization which prevents over-fitting. {Parameters: default}.
	
Decision Tree (DT)	Based on a flowchart-like tree structure. Each node is a test case, branches are the possible outcomes of the node, and terminates into a leaf (label). Breaks complex problems into simpler prob-lems by utilizing a multilevel approach. {Parameters: criterion = ‘entropy’, random_state = 0, max_depth = None}
	
Random Forest (RF)	A form of bagging, non-parametric ensemble algorithm. It is effective at increasing test accuracy while also reducing over-fitting. {Parameters: default}.
	
K-Nearest Neighbor (KNN)	Classification points are calculated using Euclidian distances to find the best K values for a given dataset. KNNs are very time-consuming. Follows a non-parametric method. {Parameters: n_neighbors = 9, metric = minkowski, p = 2}.
	
Multi-Layer Perceptron (MPL)	Multi-Layer Perceptron (MLP) also called Artificial Neural Net-work (ANN) are fully interconnected neural networks where all the neurons are interconnected with each other. All the intercon-nected layers use an activation function, usually ReLU. Output layer activated using a sigmoid function in case of binary classifi-cation. Used in applications of face recognition [27], speech recognition [28], etc.
Gaussian Naïve Bayes (GNB)	Follows a Gaussian Normal system and the Bayes Theorem where we find the probability of one event occurring given that another already happened. {Parameters: default}.
	
1-Dimensional Convolu-tional Neural Network (Conv1D)	The model consists of three Conv1D layers each paired with 1-dimensional max pooling with no padding. The output of the model is flattened and processed by 3 fully connected perceptron neural networks. Various real-world applications such as stock price prediction on daily stocks [29], pathfinding for mobile robots in unstructured environments [30] and automatically tag-ging music [31] and Mycobacterium tuberculosis detection [32]. 
	
	
	
# 5.	Experimental Results
In this section, experimental results are discussed in detail. All the machine learning models, boosting techniques and deep learning models are evaluated against stand-ard scoring metrics for a classification problem and the best mode is chosen based on the standard evaluation metrics. Metrics used are Sensitivity, Specificity, Precision, Matthews Correlation Coefficient (MCC), Negative Predicted Value (NPV), False Discovery Rate (FDR), False Negative Rate (FNR), F1 Score and Accuracy.
Keras Tuner was used to go through all the permutations of hidden layers that can be used and found the best MLP architecture. In this architecture Tensorflow implicit-ly defines the initial input layer of 18 input neurons corresponding to the feature space. The input layer is followed by 6 fully connected hidden layers with 512, 256, 128, 32, 16 and 2 neurons respectively. All the hidden layers use the ReLU activation function and the kernel initializer used is “he_uniform”. Finally, there is an output layer of 1neuron activated by the sigmoid activation function for binary classifica-tion. The model was trained over 10 epochs over a batch size of 64 and a validation split of 0.2 was used. Binary Cross-Entropy loss function and Adamax optimizer were employed.
The Conv1D architecture has 3 1d-convolutional layers of units 128, 64 and 64 respectively, followed by a MaxPool1D layer of pool size 2 and stride 2. The architec-ture is then flattened and linked with 4 fully connected layers with 3471, 847, 847 and 521 neurons all activated with ReLU. Each fully connected layer undergoes Batch Normalization and Dropout at a rate of 0.5 to prevent over-fitting. Output layers contain 1 neuron with sigmoid activation. Fig. 4. a and b show Loss vs. Epoch of MLP and Conv1D while training.


Table. 3 presents the results of all the classifiers. All the metrics were calculated using the results from the confusion matrix generated upon the test set.
Table 3. Model Performance Table
Model	Accuracy	Precision	Sensitivity	Specificity	F1 Score	MCC	CM (2x2)
LR	0.9966	0.9978	0.9978	0.9929	0.9978	0.9906	68939	149
							153	20795
Conv1D	0.9972	0.9988	0.9975	0.9962	0.9981	0.992	69004	84
							171	20777
KNN	0.9966	0.9979	0.9977	0.993	0.9925	0.9905	68942	146
							158	20790
DT	0.9961	0.9975	0.9975	0.9916	0.9975	0.9892	68912	176
							171	20777
RF	0.9972	0.9985	0.9979	0.9949	0.9982	0.9922	68981	107
							149	20799
GNB	0.9918	0.9913	0.998	0.972	0.9947	0.9774	68488	600
							134	20814
XGBoost	0.9974	0.9987	0.9979	0.9957	0.9983	0.9927	68999	89
							144	20804
MLP	0.9973	0.999	0.9975	0.9967	0.9983	0.9925	69020	68
							173	20775
              
              
MLP achieves an accuracy of 99.73% accuracy because MLP can learn complex and non-linear relations feasibly. Random Forest and Conv1D get an accuracy score of 99.72% on the test set. On the other hand, Gaussian Naïve Bayes has the lowest accuracy of 99.18% among all the models tested, although Gaussian NB had the highest Sensitivity score of 0.998 followed by XGBoost and Random Forest with a score of 0.9979. MLP takes the highest scores in Specificity and Precision of 0.9967 and 0.999 respectively. Gaussian NB has the lowest Specificity score (0.972) and the lowest Precision score of 0.9913. MLP and XGBoost have the highest F1 Score and MCC Score of 0.9983 and 0.9927 (0.9925 for MLP) respectively. Gaussian NB scores the lowest in FNR of 0.002 and the highest NPV score of 0.9936. Finally, MLP has the lowest FDR score of 0.001 followed by Conv1D. From Table. 3. it can be inferred that Conv1D is a close competitor of MLP. A confusion matrix (CM) is a table that is used to define the performance of a classification algorithm. A confusion matrix visualizes and summarizes the performance of a classification algorithm. All the metrics were derived from the given CM (2X2). FN, TP, TN, and FP stand for False Negative, True Positive, True Negative, and False Positive respectively. On test-ing the feature extraction algorithm and the classifiers on a different URL dataset. Conv1D had the highest accuracy of 92.47% followed by Logistic Regression with an accuracy of 90.31%.


# 6	Conclusion & Future Work
In this paper, lexical features were extracted from URLs and used to train various models. The empirical results show us that MLP is the best overall performing model across all metrics. This is true because of MLP’s ability to learn complex and non-linear relations easily. The techniques described in this paper can be implemented on IT security systems. Decision Tree was the worst classifier model among all the mod-els based on the standard metric results tested in this paper. 
For future work, we would develop a self-feature-extracting neural network model that could provide an end-to-end solution with high accuracy and implement recur-rent neural networks like LSTM and gated recurrent units to possibly obtain higher accuracies with low training times and computational cost. These models do not ac-count for the new font-based URL attacks. Further incorporation of font-based URLs in training data could help in the detection of this type of attack.



# References
1.	Internet Security Threat Report (ISTR) 2019–Symantec, https://www.symantec.com/content/dam/symantec/docs/reports/istr-24-2019-en.pdf, last ac-cessed 2022/3/17.
2.	Sahoo, D., Liu, C., & Hoi, S. C. (2017). Malicious URL detection using machine learning: A survey. arXiv preprint arXiv:1701.07179.
3.	Khonji, M., Iraqi, Y., & Jones, A.: Phishing detection: a literature survey. IEEE Communi-cations Surveys & Tutorials, 15(4), 2091-2121 (2013).
4.	Cova, M., Kruegel, C., & Vigna, G.: Detection and analysis of drive-by-download attacks and malicious JavaScript code. In Proceedings of the 19th international conference on World wide web (pp. 281-290) (2010).
5.	Heartfield, R., & Loukas, G.: A taxonomy of attacks and a survey of defence mechanisms for semantic social engineering attacks. ACM Computing Surveys (CSUR), 48(3), 1-39 (2015).
6.	Prakash, P., Kumar, M., Kompella, R. R., & Gupta, M.: Phishnet: predictive blacklisting to detect phishing attacks. In: 2010 Proceedings IEEE INFOCOM (pp. 1-5). IEEE (2010).
7.	Garera, S., Provos, N., Chew, M., & Rubin, A. D.: A framework for detection and meas-urement of phishing attacks. In: Proceedings of the 2007 ACM workshop on Recurring malcode (pp. 1-8) (2007).
8.	Khonji, M., Jones, A., & Iraqi, Y.: A study of feature subset evaluators and feature subset searching methods for phishing classification. In: Proceedings of the 8th annual collabora-tion, electronic messaging, anti-abuse and spam conference (pp. 135-144) (2011).
9.	Kuyama, M., Kakizaki, Y., & Sasaki, R.: Method for detecting a malicious domain by using whois and dns features. In: The third international conference on digital security and foren-sics (DigitalSec2016) (Vol. 74) (2016).
10.	Ma, J., Saul, L. K., Savage, S., & Voelker, G. M.: Learning to detect malicious urls. ACM Transactions on Intelligent Systems and Technology (TIST), 2(3), 1-24 (2011).
11.	Singh, V., Gourisaria, M. K., GM, H., Rautaray, S. S., Pandey, M., Sahni, M., ... & Espi-noza-Audelo, L. F.: Diagnosis of Intracranial Tumors via the Selective CNN Data Modeling Technique. Applied Sciences, 12(6), 2900 (2022).
12.	Das, H., Naik, B., & Behera, H. S.: Classification of diabetes mellitus disease (DMD): a da-ta mining (DM) approach. In: Progress in computing, analytics and networking (pp. 539-549). Springer, Singapore (2018).
13.	Sarah, S., Singh, V., Gourisaria, M. K., & Singh, P. K.: Retinal Disease Detection using CNN through Optical Coherence Tomography Images. In 2021 5th International Conference on Information Systems and Computer Networks (ISCON) (pp. 1-7). IEEE (2021).
14.	Panigrahi, K. P., Sahoo, A. K., & Das, H.: A cnn approach for corn leaves disease detection to support digital agricultural system. In: 2020 4th International Conference on Trends in Electronics and Informatics (ICOEI)(48184) (pp. 678-683). IEEE (2020).
15.	Chandra, S., Gourisaria, M. K., GM, H., Rautaray, S. S., Pandey, M., & Mohanty, S. N.: Semantic analysis of sentiments through web-mined twitter corpus. In CEUR Workshop Proceedings (Vol. 2786, pp. 122-135) (2021).
16.	Pramanik R., Khare S., Gourisaria M.K.: Inferring the Occurrence of Chronic Kidney Fail-ure: A Data Mining Solution. In: Gupta D., Khanna A., Kansal V., Fortino G., Hassanien A.E. (eds) Proceedings of Second Doctoral Symposium on Computational Intelligence. Ad-vances in Intelligent Systems and Computing, vol 1374. Springer, Singapore (2022).
17.	Sun, B., Akiyama, M., Yagi, T., Hatada, M., & Mori, T.: Automating URL blacklist genera-tion with similarity search approach. IEICE TRANSACTIONS on Information and Sys-tems, 99(4), 873-882 (2016).
18.	Sinha, S., Bailey, M., & Jahanian, F.: Shades of Grey: On the effectiveness of reputation-based “blacklists”. In: 2008 3rd International Conference on Malicious and Unwanted Soft-ware (MALWARE) (pp. 57-64). IEEE (2008).
19.	Ma, J., Saul, L. K., Savage, S., & Voelker, G. M.: Beyond blacklists: learning to detect mali-cious web sites from suspicious URLs. In: Proceedings of the 15th ACM SIGKDD interna-tional conference on Knowledge discovery and data mining (pp. 1245-1254) (2009).
20.	Vundavalli, V., Barsha, F., Masum, M., Shahriar, H., & Haddad, H.: Malicious URL Detec-tion Using Supervised Machine Learning Techniques. In: 13th International Conference on Security of Information and Networks (pp. 1-6) (2020).
21.	Aydin, M., Butun, I., Bicakci, K., & Baykal, N.: Using attribute-based feature selection ap-proaches and machine learning algorithms for detecting fraudulent website URLs. In: 2020 10th Annual Computing and Communication Workshop and Conference (CCWC) (pp. 0774-0779). IEEE (2020).
22.	Bharadwaj, R., Bhatia, A., Chhibbar, L. D., Tiwari, K., & Agrawal, A.: Is this URL Safe: Detection of Malicious URLs Using Global Vector for Word Representation. In: 2022 In-ternational Conference on Information Networking (ICOIN) (pp. 486-491). IEEE (2022).
23.	https://www.kaggle.com/datasets/siddharthkumar25/malicious-and-benign-urls, last accessed 2022/3/2
24.	Singh, V., Gourisaria, M. K., & Das, H.: Performance Analysis of Machine Learning Algo-rithms for Prediction of Liver Disease. In: 2021 IEEE 4th International Conference on Com-puting, Power and Communication Technologies (GUCON) (pp. 1-7). IEEE (2021).
25.	Das, H., Naik, B., & Behera, H. S.: Medical disease analysis using neuro-fuzzy with feature extraction model for classification. Informatics in Medicine Unlocked, 18, 100288 (2020).
26.	Sarah, S., Gourisaria, M. K., Khare, S., & Das, H.: Heart Disease Prediction Using Core Machine Learning Techniques—A Comparative Study. In: Advances in Data and Infor-mation Sciences (pp. 247-260). Springer, Singapore (2022).
27.	MageshKumar, C., Thiyagarajan, R., Natarajan, S. P., Arulselvi, S., & Sainarayanan, G.: Gabor features and LDA based face recognition with ANN classifier. In: 2011 International Conference on Emerging Trends in Electrical and Computer Technology (pp. 831-836). IEEE (2011).
28.	Wijoyo, S.; Wijoyo, S.: Speech recognition using linear predictive coding and artificial neu-ral network for controlling the movement of a mobile robot. In: Proceedings of the 2011 In-ternational Conference on Information and Electronics Engineering (ICIEE 2011), Bangkok, Thailand, 28–29 (2011).
29.	Jain, S., Gupta, R., & Moghe, A. A.: Stock price prediction on daily stock data using deep neural networks. In: 2018 International conference on advanced computation and telecom-munication (ICACAT) (pp. 1-13). IEEE (2018).
30.	Visca, M., Bouton, A., Powell, R., Gao, Y., & Fallah, S.: Conv1D Energy-Aware Path Planner for Mobile Robots in Unstructured Environments. In: 2021 IEEE International Con-ference on Robotics and Automation (ICRA) (pp. 2279-2285). IEEE (2021).
31.	Kim, T., Lee, J., & Nam, J.: Sample-level CNN architectures for music auto-tagging using raw waveforms. In: 2018 IEEE international conference on acoustics, speech and signal pro-cessing (ICASSP) (pp. 366-370). IEEE (2018).
32.	Singh V., Gourisaria M.K., Harshvardhan GM, Singh V.: Mycobacterium Tuberculosis De-tection Using CNN Ranking Approach. In: Gandhi T.K., Konar D., Sen B., Sharma K. (eds) Advanced Computational Paradigms and Hybrid Intelligent Computing. Advances in Intelli-gent Systems and Computing, vol 1373. Springer, Singapore (2022).


