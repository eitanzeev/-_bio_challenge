import matplotlib.pyplot as plt
import numpy as np

class PlottingAssistance():
    def __init__(self, n_folds, metrics):
        if metrics is None:
            metrics = ['r2', 'mae', 'rmse', 'loss']
        self.n_folds = n_folds
        self.metrics = metrics
        
    def epoch_performance_per_fold(self, model_results):

        folds = [str(fold_idx) + "_fold" for fold_idx in range(0,self.n_folds)]
        titles = []
        y_labels = []
        for m in self.metrics:
            titles.append(f'{m.upper()} Score per Epoch')
            y_labels.append(f'{m.upper()}')
       
        phases = ['train', 'val']

        for k in model_results.keys():
            slice_name = k

            # Set up the plotting
            num_metrics = len(self.metrics)
            fig, axs = plt.subplots(2, num_metrics, figsize=(20, 12), sharex=True)

            # Plotting loop
            for j, metric in enumerate(self.metrics):
                for i, phase in enumerate(phases):
                    ax = axs[i, j]  # Select subplot. Top row (i=0) for train, bottom row (i=1) for val
                    for fold in folds:
                        key = f'{phase}_{metric}'
                        # Check if the key and fold exist in the data dictionary
                        if key in model_results[slice_name] and fold in model_results[slice_name][key]:
                            ax.plot(model_results[slice_name][key][fold], label=f'{fold}')
                        else:
                            print(f"Data for {key} and {fold} not found.")
                    ax.set_title(f'{phase.capitalize()} {titles[j]}')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel(metric.capitalize())
                    ax.legend()
            fig.suptitle(f"Epoch Level Performance Metrics for {slice_name.title()} data slice", fontsize = 16)

        plt.tight_layout()

        plt.show()


    def fold_level_performance(self, avg_score_per_fold):
        phases = ['train', 'val']
        titles = [f'{m.upper()} Score per Fold' for m in self.metrics]

        for k, fold_data in avg_score_per_fold.items():
            slice_name = k
            fig2, axs2 = plt.subplots(2, len(self.metrics), figsize=(20, 10), sharex=True)  # Increased height for clarity

            for j, metric in enumerate(self.metrics):
                num_folds = len(next(iter(fold_data.values())))
                for i, phase in enumerate(phases):
                    ax = axs2[i, j]
                    fold_scores = [fold_data.get(f'{phase}_{metric}', {}).get(f'{f_idx}_fold', None) for f_idx in range(num_folds)]
                    fold_scores = [score for score in fold_scores if score is not None]

                    if fold_scores:  # Only plot if there are scores
                        ax.plot(fold_scores, label=f'{phase} {metric}')
                        ax.legend()  # Ensure legend is set for each subplot

                    ax.set_title(f'{phase.capitalize()} {metric.capitalize()} per Fold')
                    ax.set_xlabel('Fold')
                    ax.set_ylabel(metric.capitalize())
                    ax.set_xticks(range(num_folds))
                    ax.set_xticklabels([f'{f_idx}_fold' for f_idx in range(num_folds)])

            fig2.suptitle(f"Fold Level Performance Metrics for {slice_name.title()} Data Slice", fontsize=16)
        plt.tight_layout()
        plt.show()


