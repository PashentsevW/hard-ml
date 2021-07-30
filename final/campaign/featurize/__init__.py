from .calcers import *
from .compute import (registerCalcer,
                      compute_features,)

registerCalcer(PurchasesBaseCalcer)
registerCalcer(DayOfWeekPurchasesCalcer)
registerCalcer(AgeLocationCalcer)
registerCalcer(TargetFromCampaignCalcer)

