

class TemplateMetric():
    """
    Abstract template for metric
    """
    def __init__(self):
        pass        

    def compute(self, output, target):
        raise NotImplementedError("This is an abtract method")

    def update(self,  output, target):
        raise NotImplementedError("This is an abtract method")


    def reset(self):
        raise NotImplementedError("This is an abtract method")


    def value(self):
        raise NotImplementedError("This is an abtract method")


    def __str__(self):
        raise NotImplementedError("This is an abtract method")

    def __len__(self):
        raise NotImplementedError("This is an abtract method")


    
  
