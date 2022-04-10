import numpy

class ClassWeights():
    def __init__(self, y, numClass, strategy, beta=0.9, normalise=False):
        self.weights = None
        if strategy == 'ENS':
            self.weights = self.get_class_weights_ENS(y, numClass, beta=beta, normalise=normalise)
        if strategy == 'INS':
            self.weights = self.get_class_weights_INS(y, numClass, beta=beta, normalise=normalise)
        if strategy == 'ISNS':
            self.weights = self.get_class_weights_ISNS(y, numClass, beta=beta, normalise=normalise)
    def __call__(self):
        return self.weights

    def get_class_weights_ENS(self, y, numclass, beta=0.9, normalise=False):
        """
        Get class weights for imbalanced dataset
        """
        effectiveNum = 1.0 - numpy.power(beta, y)
        weights = (1 - beta) / numpy.array(effectiveNum)
        if normalise:
            weights /= (numpy.sum(weights) * numclass)
        return weights

    def get_class_weights_INS(self, y, numclass, beta=0.9, normalise=False):
        """
        Get class weights for imbalanced dataset
        """
        weights = 1 / (y)
        if normalise:
            weights /= (numpy.sum(weights) * numclass)
        return weights    

    def get_class_weights_ISNS(self, y, numclass, beta=0.9, normalise=False):
        """
        Get class weights for imbalanced dataset
        """
        weights = 1 / (numpy.sqrt(y))
        if normalise:
            weights /= (numpy.sum(weights) * numclass)
        return weights    
