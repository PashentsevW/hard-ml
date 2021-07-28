from .calcers import *
from .compute import (registerCalcer,
                      compute_features,)

registerCalcer(PurchasesAggregateCalcer)
registerCalcer(AgeLocationCalcer)

