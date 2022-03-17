from torch.utils.data import DataLoader, random_split,Dataset

class HousePriceLoader : 
    def __init__(self, dataset:Dataset) -> None:
        """ Create train test split"""
        self.__loader =  DataLoader(
            dataset, 
            batch_size=200, 
            shuffle=True
        )
        
    def get_loader(self)->DataLoader : 
        """Getter for train loader"""
        return self.__loader