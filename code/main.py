from dataset import HousePricesDataSet
from dataloader import HousePriceLoader
from feature_engg import feature_engg
import config
from model import QuantModel
from pytorch_lightning import LightningDataModule, LightningModule, Trainer


def main()-> None: 
    """Main function"""
    # Create Datset
    dataset = HousePricesDataSet(
        train_data_dir=config.TRAIN_TEST_DIR,
        feature_engg_func=feature_engg,
    )
    dataloader = HousePriceLoader(
        dataset=dataset,
        train_size=config.TRAIN_SIZE
    )
    # Train the model
    trainer = Trainer(max_epochs=100,)
    quant_model = QuantModel(
        num_features=289, # Num of input features set in the dataset function
        quantiles=config.QUANTILES
    )
    trainer.fit(
        model = quant_model, 
        train_dataloader= dataloader.get_trainloader(), 
        val_dataloaders= dataloader.get_valloader()
    )






if "__main__" == __name__ : 
    main()
    print("complete")