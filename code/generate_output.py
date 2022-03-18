from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from model import QuantModel
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import shap
import sklearn
import numpy as np


class GenerateOutput : 
    """Calls to generate predictions and shap values"""
    def __init__(
        self, 
        dataloader :DataLoader, 
        trainer:Trainer, 
        model:QuantModel, 
        actuals:List[float],
        train_dataloader:DataLoader,
        feature_names : List[str] , 
        mode:str,
    ) -> None:
        self.dataloader = dataloader
        self.trainer = trainer
        self.model = model
        self.actuals = actuals
        self.preds=None
        self.train_dataloader = train_dataloader
        self.feature_names = feature_names
        self.mode=mode

    def run_test_step(self) : 
        """Run the test step"""
        self.trainer.test(dataloaders=self.dataloader)

    def __print_metrics(self, actual:List[float], predicted:List[float]) : 
        def smape(A, F):
            return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

        smape_value = smape(actual, predicted)
        mae_value = sklearn.metrics.mean_absolute_error(actual, predicted)
        mape_val = sklearn.metrics.mean_absolute_percentage_error(actual, predicted)
        rmse = sklearn.metrics.mean_squared_error(actual, predicted, squared=False)
        print(f'The metrics for {self.mode} are : \nMAE : {mae_value}\nMAPE: {mape_val}\nSMAPE : {smape_value}\nRMSE : {rmse}')

    def __generate_predictions(self):
        """Run shap values"""
        all_preds=[]
        for batch_idx, batch in enumerate(self.dataloader):
            pred = self.model.predict_step(batch, batch_idx)
            all_preds.append(pred)

        # predictions
        preds = pd.DataFrame(
            all_preds[0].detach().numpy(), 
            columns=["ub", "mean", "lb"]
        )
        #
        preds['actuals'] = self.actuals
        self.preds = preds.sort_values(by='actuals').reset_index(drop=True)
        #
        self.__print_metrics(predicted=self.preds['mean'], actual=self.preds.actuals)

    def explain_shap(self):
        """Explain shap for a few predictions"""
        
        # Get data loaders
        batch = next(iter(self.dataloader))
        images, _ = batch

        batch = next(iter(self.train_dataloader))
        background, _ = batch
        
        # shap
        explainer = shap.DeepExplainer(
            model = self.model, 
            data = background # data.val.drop(columns=['SalePrice']).head(50)
        )
        shap_values = explainer.shap_values(images[:5])
        plt.clf()
        shap.summary_plot(
            shap_values[0], 
            plot_type = 'bar', 
            feature_names = self.feature_names,
            max_display=10,
            show=False
        )
        plt.savefig(f"output/SHAP_{self.mode}.png",bbox_inches='tight')
    
    def plot_predictions(self):
        """Get the plot """
        self.__generate_predictions()
        self.__plot_mean_and_CI(
            actuals = self.preds['actuals'], 
            mean = self.preds['mean'] , 
            lb = self.preds[f'lb'] ,
            ub = self.preds[f'ub'] , 
            color_mean='r', 
            color_shading='r', 
            color_actuals='g', 
        )

    def __plot_mean_and_CI(self,
        actuals, mean, lb, ub, color_mean=None, 
        color_shading=None, color_actuals=None
    ):
        plt.clf()
        fig = plt.figure(1, figsize=(20, 6))
        # plot the shaded range of the confidence intervals
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
        plt.savefig(f"output/CI_ACT_PRED__{self.mode}.png",bbox_inches='tight')
        
