import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import confusion_matrix
class KNNClassifier:
    def __init__(self,X,y,K):
        #initialize all attributes
        self.X = np.array(X) #2d-array
        self.y = np.array(y)
        self.n_observations = len(y) #number of labels
        self.classes = np.unique(self.y) 
        self.K = K
    def predict(self,X):
        X = np.array(X)
        predictions = []
        for i in range(X.shape[0]):
            distances = np.sqrt(np.sum((X[i,:]-self.X)**2,axis=1))
            idx = np.argsort(distances)[:self.K]
            knn_labels = self.y[idx]
            knn_distances = distances[idx]
            best_dist = 0
            best_class = 0
            best_count = 0
        
            for label in self.classes:
            #for j in range(len(self.classes)): #for label in slef.classes
                temp_count = np.sum(label==knn_labels)
                #temp_count = np.sum(label == knn_labels)
                temp_dist = np.sum(knn_distances[knn_labels==label])
            
                if(temp_count > best_count):
                    best_dist = np.sum(knn_distances[knn_labels==label])
                    best_class = label
                    best_count = np.sum(knn_labels==label)
                elif ((temp_count == best_count) & (temp_dist < best_dist)):
                    best_dist = np.sum(knn_distances[knn_labels==label])
                    best_class = label
                    best_count = np.sum(knn_labels==label)
                
            predictions.append(best_class)
        
        predictions = np.array(predictions)
        return predictions
        
        
        
    def score(self,X,y):
        #correct x and y to numpy array
        X = np.array(X)
        y_label = np.array(y)
        number_y = len(y_label)
        #calsulate y_predicted
        y_predicted = self.predict(X)     
        #calculate accuracy
        accuracy = (np.sum(y_label==y_predicted))/(number_y)      
        return accuracy
        
    def confusion_matrix(self,X,y):
    #TP FP TN FN
        y_predicted = self.predict(X)
        co = confusion_matrix(y,y_predicted)
        return co
 
    
#test code
np.random.seed(1204)   
X = np.random.uniform(0,10,40).reshape(20,2)
y = np.random.choice(['a','b','c','d'],20)
Knn_mod = KNNClassifier(X,y,3)
print(Knn_mod.score(X,y))
print(Knn_mod.predict(X))
print(Knn_mod.confusion_matrix(X,y))
