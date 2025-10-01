import matplotlib.pyplot as plt
import seaborn as sns

class ManufacturingVisualizer:
    def __init__(self, df):
        self.df = df

    def plot_correlation(self):
        plt.figure(figsize=(8,6))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.show()

    def plot_trend(self, time_col, target_col):
        trend = self.df.groupby(time_col)[target_col].mean()
        plt.figure(figsize=(8,4))
        trend.plot(marker='o')
        plt.title(f"Trend of {target_col} over {time_col}")
        plt.show()

    def plot_spc(self, target_col, mean, ucl, lcl):
        x = self.df[target_col].dropna().reset_index(drop=True)
        plt.figure(figsize=(10,4))
        plt.plot(x, marker='o')
        plt.axhline(mean, color='green', linestyle='--', label='Mean')
        plt.axhline(ucl, color='red', linestyle='--', label='UCL')
        plt.axhline(lcl, color='red', linestyle='--', label='LCL')
        plt.legend()
        plt.title(f"SPC Chart - {target_col}")
        plt.show()
