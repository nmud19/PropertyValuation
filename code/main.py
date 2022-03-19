from generate_output import GenerateOutput
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


def main() -> None:
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
        max_epochs=2000,
    )
    quant_model = QuantModel(
        num_features=285,  # Num of input features set in the dataset function
        quantiles=config.QUANTILES,
    )
    trainer.fit(
        model=quant_model,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    generate_output = GenerateOutput(
        dataloader=val_dataloader,
        trainer=trainer,
        model=quant_model,
        actuals=data.val.head(200).SalePrice.values,
        train_dataloader=train_dataloader,
        feature_names=data.val.drop(columns=["SalePrice"]).columns,
        mode="val",
    )
    generate_output.plot_predictions()
    generate_output.explain_shap()

    #
    generate_output = GenerateOutput(
        dataloader=test_dataloader,
        trainer=trainer,
        model=quant_model,
        actuals=data.te.head(200).SalePrice.values,
        train_dataloader=train_dataloader,
        feature_names=data.te.drop(columns=["SalePrice"]).columns,
        mode="test",
    )
    generate_output.plot_predictions()
    generate_output.explain_shap()

    # shap.initjs()
    # shap.force_plot(
    #     explainer.expected_value[0],
    #     shap_values[0][0],
    #     features = data.val.drop(columns=['SalePrice']).columns
    # )

    # shap.plots._waterfall.waterfall_legacy(
    #     explainer.expected_value[0],
    #     shap_values[0][0],
    #     feature_names = data.val.drop(columns=['SalePrice']).columns
    # )


if "__main__" == __name__:
    main()
    print("complete")
