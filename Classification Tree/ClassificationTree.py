#Alexandru Andrei
import numpy as np

class ClassificationTree:
    def __init__(self, X, y, max_depth=2, depth=0,min_leaf_size=2,classes=None):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_bos = len(y)
        self.depth = depth
        self.class_count = []
        if classes is None:
            self.classes = np.unique(self.y)
        else:
            self.classes = classes
  
        for label in self.classes:
            self.class_count.append(np.sum(label==self.y))
            
        self.class_count = np.array(self.class_count)
        self.prediction = self.classes[np.argmax(self.class_count)] 
        class_ratios = self.class_count / self.n_bos
        self.gini = 1 - np.sum(class_ratios **2)
        
        
        if (depth == max_depth) or ( self.gini == 0):
            self.axis = None
            self.t = None
            self.left = None
            self.right = None
            return
        
        self.axis = 0
        self.t = 0
        split_gini = 1
        X_val = np.array(self.X)

        for item in range(X_val.shape[1]): 
            col_values = X_val[:,item].copy()
            col_values = np.sort(col_values) 
                
            for row in range (len(col_values)):
                sel = X_val[:,item] <= col_values[row]
                _,l_counts = np.unique(self.y[sel], return_counts=True) 
                _,r_counts = np.unique(self.y[~sel], return_counts=True)
                n_left = np.sum(sel)
                n_right = np.sum(~sel)
                if n_left == 0 :
                    l_gini = 1
                else:
                    l_gini = 1-np.sum((l_counts/n_left)**2)
                    
                if n_right == 0:
                    r_gini =1
                else:
                    r_gini = 1-np.sum((r_counts/n_right)**2)
                    
                gini = (n_left * l_gini + n_right * r_gini) / (n_left + n_right)

                if (gini <= split_gini) & (n_left >= min_leaf_size) & (n_right >= min_leaf_size):
                    self.axis = item
                    if((row +1)== len(col_values)):
                        self.t = col_values[row]
                    else:
                        self.t = (col_values[row]+col_values[row+1])/2
                    split_gini = gini
        sel = X_val[:,self.axis]<=self.t
        if (np.sum(sel) < min_leaf_size) or (np.sum(~sel) < min_leaf_size):
                  self.left = None
                  self.right = None
                  self.axis = None
                  self.t = None
                  return

        self.left = ClassificationTree(X[sel,:], y[sel], max_depth, depth+1, min_leaf_size,self.classes)
        self.right = ClassificationTree(X[~sel,:], y[~sel], max_depth, depth+1, min_leaf_size,self.classes)
         
                
    def classify_row(self,row):
        row = np.array(row)
        
        if self.left == None or self.right == None:
            return self.prediction 
    
        if row[self.axis] <= self.t: #left
            return self.left.classify_row(row)
        else:
            return self.right.classify_row(row)           
        
    def predict(self,X):
        X = np.array(X)
        predictions = []
        
        for i in range(X.shape[0]):
            row = X[i,:]
            predictions.append(self.classify_row(row))
            
        predictions = np.array(predictions)
        return predictions
    
    
    def score(self,X,y):
        X_value = np.array(X)
        y_label = np.array(y)
        number_y = len(y_label)
        y_predicted = self.predict(X_value)     
        accuracy = (np.sum(y_label==y_predicted))/(number_y)      
        return accuracy
        
        
    def print_tree(self):
        msg = '  ' * self.depth + '* Size = ' + str(self.n_bos)+' '+str(self.class_count)
        msg += ', Gini: ' + str(round(self.gini,2))           
        if(self.left != None):
            msg += ', Axis: ' + str(self.axis)
            msg += ', Cut: ' + str(round(self.t,2))
        else:
            msg += ', Predicted Class: ' + str(self.prediction )
                
        
        print(msg)
        
        if self.left != None:
            self.left.print_tree()
            self.right.print_tree()