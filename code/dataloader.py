from torch.utils.data import DataLoader, random_split,Dataset

class HousePriceLoader : 
    def __init__(self, dataset:Dataset, train_size:float=0.8, ) -> None:
        """ Create train test split"""
        train_size = int(train_size * len(dataset))
        valid_size = len(dataset) - train_size
        trainset, validationset = random_split(dataset, [train_size, valid_size])
        self.__trainloader =  DataLoader(trainset, batch_size=200, shuffle=True)
        self.__validationloader = DataLoader(validationset, batch_size=200, shuffle=False)

    def get_trainloader(self)->DataLoader : 
        """Getter for train loader"""
        return self.__trainloader

    def get_valloader(self)->DataLoader : 
        """Getter for train loader"""
        return self.__validationloader
    