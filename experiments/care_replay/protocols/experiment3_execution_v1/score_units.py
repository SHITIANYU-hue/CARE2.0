"""User-approved source-only score-unit adapter.
See approved_score_units_amendment.json for authorization.
"""
from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class SourceScoreUnits:
    source_median: float
    source_iqr: float
    source_count: int
    @classmethod
    def fit(cls,source_fit_values):
        x=np.asarray(source_fit_values,dtype=np.float64)
        if x.ndim!=1 or len(x)!=1024:raise ValueError('Expected exactly1024 frozen source-fit outcomes')
        if not np.isfinite(x).all():raise ValueError('Invalid source outcome')
        q25,median,q75=np.quantile(x,[.25,.5,.75],method='linear');iqr=float(q75-q25)
        if not np.isfinite(iqr) or iqr<=0:raise ValueError('Nonpositive source IQR; existing exclusion policy applies')
        return cls(float(median),iqr,len(x))
    def revealed_to_executor(self,revealed_values):
        x=np.asarray(revealed_values,dtype=np.float64)
        if not np.isfinite(x).all():raise ValueError('Invalid revealed outcome')
        result=100*((x-self.source_median)/self.source_iqr)
        if not np.isfinite(result).all():raise ValueError('Nonfinite transformed outcome')
        return result
    def executor_auc_delta_to_source_iqr(self,delta):return np.asarray(delta,dtype=np.float64)/100.
