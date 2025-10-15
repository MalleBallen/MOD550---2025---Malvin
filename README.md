# MOD550---2025---Malvin
A repository in the course MOD550 - Fundaments of Machine Learning for and with Engineering Applications. 

*Machine learning has recently emerged as one of the most promising resources for engineers, offering a set of powerful approaches to tackle complex engineering challenges. By employing various statistical techniques in learning algorithms, machine learning enables the development of predictive models, optimization strategies, and decision support systems that can enhance the design, analysis, and control of engineered systems. This technology empowers engineers to extract meaningful insights from vast datasets, automate repetitive tasks, and improve the efficiency, reliability, and performance of engineering processes. Its applications span diverse domains, from mechanical and chemical engineering to geo, material science, etc. making machine learning an indispensable resource for modern computational engineers for physics based and data-driven solutions to real-world problems.*  

 

*Most undergraduates in engineering and science fields have little exposure to data methods, while most computer scientists and statisticians have little exposure to dynamical system control. Our goal is to provide an entry point and interface for both these groups of students.*  

****About me****  
I'm a 25 year old from Arendal, currently living in Stavanger with my girlfriend. In addition to having a full-time job as a purchaser in Aibel, I love a good challenge. That's why I'm doing my best to complete the master course on UiS in computational engineering alongside work. It might be a bit over my head, but at least I will give it my best shot :D  
*****Warning! Vital info ahead!*****  
I have a 12 week old golden retriever (As of middle of september). This might or might not save lives. 🐶

**Task 1**  

1: make a histogram from a 2d random distribution  
2: make a 2d heat map from a 2d random distribution  
3: make a histogram for the source data you selected  
4: convert the histogram into a discrete PMF  
5: calculate the cumulative for each feature  
  
**Task 2**

1: Make a DataModel class that reads the output of the DataAquisition class (from task1) in its __init__()  
2: Make a function in DataModel to make a linear regression. I suggest you try to do it on your own with only vanilla python and the class notes. If you are lost, you can find practical info here on how to use already made libraries: https://www.geeksforgeeks.org/machine-learning/regularization-in-machine-learning/Lenker til en ekstern side.  The issue here will be data structure: np.array vs list of list vs pandas DataFrames.  
3: Make a function that split the data you got from DataAquisition into train, validation and test. Do it  with vanilla python. You need to make sure you understand the data structure.  
4: Make a function that computes MSE (make your own, don't copy from my notes :P )  
5: Make a function to make NN. It would be essentially a wrapper of other libraries, I suggest to use Keras:  https://www.geeksforgeeks.org/machine-learning/how-to-create-models-in-keras/  . You should have acquired enough notions to handle this tool.  
6: Make a function that does K_MEAN and GMM (we will discuss them next week)  
Once these methods (recipes) are done, you can now make cakes! :) :  
7: Make a linear regression on all your data (statistic ).  
8: Make a linear regression on all your train data and test it on your validation.  
9: Compute the MSE on your validation data.   
10: Try for different distribution of initial data point, (a) Discuss how different functions can be used in the linear regression, and different NN architecture. (b) Discuss how you can use the validation data for the different cases. (c) Discuss the different outcome from the different models when using the full dataset to train and when you use a different ML approach. (d) Discuss the outcomes you get for K-means and GMM. (e) Discuss how you can integrate supervised and unsupervised methods for your case.