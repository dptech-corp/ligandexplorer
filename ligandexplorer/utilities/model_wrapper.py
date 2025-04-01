class ModelWrapper:  
    def __init__(self, model, label_encoder):  
        self.model = model  
        self.le = label_encoder  
    
    def predict(self, X):  
        numeric_pred = self.model.predict(X)  
        return self.le.inverse_transform(numeric_pred)  
    
    def predict_proba(self, X):  
        return self.model.predict_proba(X)  