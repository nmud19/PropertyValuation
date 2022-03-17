from dataset import HousePricesDataSet
from dataloader import HousePriceLoader
from feature_engg import feature_engg
import config
from model import QuantModel
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from create_train_test_val import CreateData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main()-> None: 
    """Main function"""

    # Create Data
    data = CreateData(df_path=config.TRAIN_TEST_DIR)

    # Create Datset
    train_dataloader = HousePriceLoader(
        dataset=HousePricesDataSet(df=data.tr)
    ).get_loader()
    val_dataloader = HousePriceLoader(
        dataset=HousePricesDataSet(df=data.val)
    ).get_loader()
    test_dataloader = HousePriceLoader(
        dataset=HousePricesDataSet(df=data.te)
    ).get_loader()

    # Train the model
    trainer = Trainer(
        max_epochs=1000,
    )
    quant_model = QuantModel(
        num_features=285, # Num of input features set in the dataset function
        quantiles=config.QUANTILES
    )
    trainer.fit(
        model = quant_model, 
        train_dataloader= train_dataloader, 
        val_dataloaders= val_dataloader,
    )

    # test step 
    trainer.test(dataloaders=test_dataloader)

    # Generate individual preds
    all_preds=[]
    for batch_idx, batch in enumerate(test_dataloader):
        pred = quant_model.predict_step(batch, batch_idx)
        all_preds.append(pred)

    # predictions
    preds = pd.DataFrame(
        all_preds[0].detach().numpy(), 
        columns=["ub", "mean", "lb"]
    )
    for x in preds : 
        preds[x] = np.expm1(preds[x])
    
    preds['actuals']=np.expm1(data.te.SalePrice.values)
    print()

    def plot_mean_and_CI(actuals, mean, lb, ub, color_mean=None, 
                     color_shading=None, color_actuals=None):
        # plot the shaded range of the confidence intervals
        plt.clf()
        plt.fill_between(
            range(mean.shape[0]), 
            ub, lb,
            color=color_shading, 
            alpha=.1, 
            label = "{} % region".format( int(0.95*100))
        )
        # plot the mean on top
        plt.plot(mean, color_mean, label = "Predictions")
        plt.plot(actuals, color_actuals, label = "Actuals")
        plt.legend()
        plt.show()
        

    fig = plt.figure(1, figsize=(20, 6))
    plot_mean_and_CI(
        actuals = preds['actuals'], 
        mean = preds['mean'] , 
        lb = preds[f'lb'] ,
        ub = preds[f'ub'] , 
        color_mean='r', 
        color_shading='r', 
        color_actuals='g', 
    )


if "__main__" == __name__ : 
    main()
    print("complete")