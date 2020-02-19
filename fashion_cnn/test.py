class FashionDataset:
    

    def __init__(self):
        
        self.class_dict={1:'short sleeve top',2:'long sleeve top',3:'short sleeve outwear',4:'long sleeve outwear',5:'vest',6:'sling',\
                        7:'shorts',8:'trousers',9:'skirt',10:'short sleeve dress',11:'long sleeve dress',12:'vest dress',13:'sling dress'}


    def print_class(self):
        print(self.class_dict)



a=FashionDataset()
a.print_class()