Purpose: Compare Support Vector Machine (SVM), Random Forest, K-nearest neighbor, and K-Means Clustering on their accuracy in detecting data anomalies in the Numenta Anomaly Benchmark artificial dataset. The Numenta Anomaly Benchmark Artificial dataset can be accessed here: https://github.com/numenta/NAB/tree/master/data.

Description: I compared different data window sizes and tested these algorithms on their accuracy, precision, recall, AUC-ROC, AUC-PR, false positives, and false negatives on detecting outliers and tested on detecting differences in waveform patterns. I used 80% of each clean outlier dataset for training and the other 20% for testing. I split the four datasets with an anomaly in pattern, with two for training and two for testing.

Here are the results from this code:
![Window_size](https://github.com/user-attachments/assets/0bde2fc7-80a6-46e5-9245-ca8bf57bd0bb)
![All_window_size](https://github.com/user-attachments/assets/dfd8280d-96bf-4aee-b962-69be2e7b8253)
![image](https://github.com/user-attachments/assets/16b1b1a0-7d10-4466-8611-520fee503c0d)

Instructions on how to replicate my result:
I graphed the whole dataset in dataset.py and graphed a window of the four datasets with an anomaly in pattern in graphing.py. The comparison of the machine learning (ML) algorithms is in both outliers.py and main.py. outliers.py only includes outlier data anomalies, while main.py include both outliers and pattern changes. 

References:
[1]  Simon Duque Anton, Suneetha Kanoor, Daniel Fraunholz, and Hans Dieter Schotten. 2018. Evaluation of Machine  Learning-based Anomaly Detection Algorithms on an Industrial Modbus/TCP Data Set. In Proceedings of the 13th International Conference on Availability, Reliability and Security (ARES '18). Association for Computing Machinery, New York, NY, USA, Article 41, 1–9. https://doi-org.portal.lib.fit.edu/10.1145/3230833.3232818

[2]  Priyanka More, Dharmesh Dhabliya, Jambi Ratna Raja Kumar, Supriya Arvind Bhosale, Aarti S. Gaikwad, and Sonu V. Khapekar. 2024. Utilizing Machine Learning Approaches for Anomaly Detection in Industrial Control Systems. In Proceedings of the 5th International Conference on Information Management & Machine Intelligence (ICIMMI '23). Association for Computing Machinery, New York, NY, USA, Article 101, 1–7. https://doi-org.portal.lib.fit.edu/10.1145/3647444.3647928
