import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# This  file is to plot and save the dices and losses as subplots in a single files

# function to load all metrics
def load_metrics(parent_dir, stage_name='BCE', metric_name='dice_score.npy'):
    metric_dict = {}
    for subfolder in os.listdir(parent_dir):
        subfolder_path = os.path.join(parent_dir, subfolder)
        if os.path.isdir(subfolder_path):
            metric_path = os.path.join(subfolder_path, 'metrics', stage_name, metric_name)
            if os.path.exists(metric_path):
                try:
                    data = np.load(metric_path)
                    metric_dict[subfolder] = data
                except Exception as e:
                    print(f"Error loading {metric_path}: {e}")
            else:
                print(f"File not found: {metric_path}")
    return metric_dict



def plot_training_metrics(main_folder, datasets, stages, metrics, expected_epochs, load_metrics, structure):
    """
    Loads training/validation metrics, builds a DataFrame, and plots them.

    Args:
        main_folder (str): Path to the folder containing metric files.
        datasets (list): List of dataset names.
        stages (list): List of stage names.
        metrics (list): List of metric keys (e.g., ['train_dice', 'val_dice', 'train_loss', 'val_loss']).
        expected_epochs (int): Total number of expected training epochs.
        load_metrics (function): Function to load metrics. Should accept (folder, stage, filename) and return a dict.
    """
    all_data = []
    final_epochs_dict = {}  

    for dataset in datasets:
        for stage in stages:
            try:
                # Load metrics
                train_dice = load_metrics(main_folder, stage, 'train_dice.npy').get(dataset)
                val_dice = load_metrics(main_folder, stage, 'val_dice.npy').get(dataset)
                train_loss = load_metrics(main_folder, stage, 'train_loss.npy').get(dataset)
                val_loss = load_metrics(main_folder, stage, 'val_loss.npy').get(dataset)

                # Skip if any metric is missing
                if any(m is None for m in [train_dice, val_dice, train_loss, val_loss]):
                    print(f"Skipping {dataset}-{stage}: Missing one or more metrics")
                    continue

                # Ensure all arrays have same length
                lengths = list(map(len, [train_dice, val_dice, train_loss, val_loss]))
                if len(set(lengths)) != 1:
                    print(f"Skipping {dataset}-{stage}: Metric arrays have inconsistent lengths {lengths}")
                    continue

                n_epochs = lengths[0]
                final_epochs_dict[(dataset, stage)] = n_epochs

                for epoch in range(n_epochs):
                    all_data.append([
                        epoch, dataset, stage,
                        train_dice[epoch],
                        val_dice[epoch],
                        train_loss[epoch],
                        val_loss[epoch]
                    ])

            except Exception as e:
                print(f"Error processing {dataset}-{stage}: {e}")

    # Create DataFrame
    df = pd.DataFrame(
        all_data,
        columns=['epoch', 'dataset', 'stage', 'train_dice', 'val_dice', 'train_loss', 'val_loss']
    )

    # Melt for plotting
    df_melted = df.melt(
        id_vars=['epoch', 'dataset', 'stage'],
        value_vars=metrics,
        var_name='metric', value_name='value'
    )

    df_melted['type'] = df_melted['metric'].apply(lambda x: 'Train' if 'train' in x else 'Validation')
    df_melted['metric_name'] = df_melted['metric'].apply(lambda x: 'Dice' if 'dice' in x else 'Loss')

    # Plotting
    for metric_type in ['Dice', 'Loss']:
        g = sns.FacetGrid(
            df_melted[df_melted['metric_name'] == metric_type],
            col='dataset', row='stage',
            margin_titles=True, height=2.5, aspect=1.5,
            sharey=False
        )
        g.map_dataframe(
            sns.lineplot, x='epoch', y='value', hue='type', palette='Set1'
        )

        # Vertical line for early stopping
        for (row_val, col_val), ax in g.axes_dict.items():
            final_epoch = final_epochs_dict.get((col_val, row_val))
            if final_epoch is not None and final_epoch < expected_epochs:
                ax.axvline(
                    x=final_epoch - 1,
                    color='black',
                    linestyle='--',
                    linewidth=1,
                    label='Early Stopping'
                )

        g.add_legend()
        g.set_titles(row_template='{row_name}', col_template='{col_name}')
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(f'{metric_type}')
        save_path = os.path.join(main_folder, f"{metric_type.lower()}_{structure}.png")
        g.savefig(save_path, dpi=300)
        plt.show()
        
def save_final_metrics_to_csv(main_folder, datasets, stages, metric_names, load_metrics, structure):
    """
    Loads final evaluation metrics and saves them into a CSV file.

    Args:
        main_folder (str): Path to the folder containing metric files.
        datasets (list): List of dataset names.
        stages (list): List of stage names.
        metric_names (list): List of metric filenames (e.g., ['acc.npy', 'prec.npy']).
        load_metrics (function): Function to load metrics. Should accept (folder, stage, filename) and return a dict.
        structure (str): Name used in the output filename.
    """
    all_data = []

    for dataset in datasets:
        for stage in stages:
            metric_values = {}
            missing = False

            for metric_file in metric_names:
                metric_key = os.path.splitext(metric_file)[0]  # 'acc.npy' -> 'acc'
                value = load_metrics(main_folder, stage, metric_file).get(dataset)

                if value is None:
                    print(f"Missing {metric_key} for {dataset}-{stage}")
                    missing = True
                    break

                # Ensure value is a scalar (float/int), not array
                if isinstance(value, np.ndarray):
                    if value.size != 1:
                        print(f"Unexpected shape for {metric_key} in {dataset}-{stage}: {value.shape}")
                        missing = True
                        break
                    value = value.item()

                metric_values[metric_key] = value

            if not missing:
                row = {'dataset': dataset, 'stage': stage}
                row.update(metric_values)
                all_data.append(row)

    # Save to CSV
    df = pd.DataFrame(all_data)
    save_path = os.path.join(main_folder, f"metrics_{structure}.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved final metrics to: {save_path}")

#%%
# Main configuration
main_folder = r"..\outputs"
stages = ['BCE', 'Tversky', 'FocalTversky']
metrics = ['train_dice', 'val_dice', 'train_loss', 'val_loss']
metrics2 = ['acc.npy', 'prec.npy', 'rec.npy', 'spec.npy']
vessels_datasets = ['CHASE', 'DRIVE', 'FIVES', 'HRF', 'STARE']
OD_datasets = ['DRIONS-DB', 'DRISHTI_OD', 'REFUGE_OD']
OD_cup_datasets = ['DRISHTI_OD_cup', 'REFUGE_OD_cup']
#%%

plot_training_metrics(
    main_folder=main_folder,
    datasets=vessels_datasets,
    stages=stages,
    metrics=['train_dice', 'val_dice', 'train_loss', 'val_loss'],
    expected_epochs=50,
    load_metrics=load_metrics,
    structure="vessels"
)

plot_training_metrics(
    main_folder=main_folder,
    datasets=OD_datasets,
    stages=stages,
    metrics=['train_dice', 'val_dice', 'train_loss', 'val_loss'],
    expected_epochs=50,
    load_metrics=load_metrics,
    structure="OD"
)

plot_training_metrics(
    main_folder=main_folder,
    datasets=OD_cup_datasets,
    stages=stages,
    metrics=['train_dice', 'val_dice', 'train_loss', 'val_loss'],
    expected_epochs=50,
    load_metrics=load_metrics,
    structure="OD_cup"
)


#%%
save_final_metrics_to_csv(main_folder=main_folder,
                          datasets=vessels_datasets,
                          stages=stages,
                          metric_names=metrics2,
                          load_metrics=load_metrics,
                          structure='vessels')

save_final_metrics_to_csv(main_folder=main_folder,
                          datasets=OD_datasets,
                          stages=stages,
                          metric_names=metrics2,
                          load_metrics=load_metrics,
                          structure='OD')
save_final_metrics_to_csv(main_folder=main_folder,
                          datasets=OD_cup_datasets,
                          stages=stages,
                          metric_names=metrics2,
                          load_metrics=load_metrics,
                          structure='OD_cup')

