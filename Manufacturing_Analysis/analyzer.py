class ManufacturingAnalyzer:
    def __init__(self, df):
        self.df = df

    def summary_stats(self, columns=None):
        """기본 기술통계"""
        if columns is None:
            columns = self.df.select_dtypes(include='number').columns
        return self.df[columns].describe()

    def correlation_matrix(self):
        """상관계수 분석"""
        return self.df.corr()

    def trend_analysis(self, time_col, target_col):
        """시간별 추세 분석"""
        return self.df.groupby(time_col)[target_col].mean()

    def spc_chart(self, target_col):
        """SPC 관리도용 중심선, 관리한계"""
        x = self.df[target_col].dropna()
        mean = x.mean()
        std = x.std()
        ucl = mean + 3 * std
        lcl = mean - 3 * std
        return {"mean": mean, "ucl": ucl, "lcl": lcl}

    def failure_distribution(self, target_col):
        """불량/정상 분포"""
        return self.df[target_col].value_counts(normalize=True)
